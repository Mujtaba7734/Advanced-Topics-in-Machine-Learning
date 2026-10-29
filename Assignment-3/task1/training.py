import json
import gc
import os
import math
import shutil
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from common.seed import set_seed, rng_state, restore_rng
from common.checkpoint import atomic_save, atomic_latest, load_checked
from task1.dataset import load_selected, collate
from task1.model import load_model, trainable_parameters, adapter_state, load_adapter_state, projected_adapter_state
from task1.l2sp import CPUReference
from task1.optim import build_optimizer


def token_loss_sum(logits, labels):
    targets = labels[:, 1:]
    valid = targets != -100
    return F.cross_entropy(logits[:, :-1][valid].float(), targets[valid], reduction="sum")


def response_loss_sum(model, input_ids, attention_mask, labels, chunk_size=128):
    """Equivalent to full-logit CE, without materializing masked prompt logits."""
    hidden = model.model(input_ids=input_ids, attention_mask=attention_mask,
                         use_cache=False).last_hidden_state
    targets = labels[:, 1:]
    selected = hidden[:, :-1][targets != -100]
    targets = targets[targets != -100]
    if targets.numel() == 0:
        raise ValueError("Effective microbatch has no assistant response tokens")
    losses = []
    for start in range(0, len(targets), chunk_size):
        logits = model.lm_head(selected[start:start + chunk_size])
        losses.append(F.cross_entropy(logits.float(), targets[start:start + chunk_size], reduction="sum"))
    return torch.stack(losses).sum()


def weight_decay_groups(model, decay):
    yes, no = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no if name.endswith("bias") or p.ndim < 2 else yes).append(p)
    return [{"params": yes, "weight_decay": decay}, {"params": no, "weight_decay": 0.0}]


def trim_jsonl_to_checkpoint(path, field, maximum):
    lines = path.read_text(encoding="utf-8").splitlines()
    retained = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break  # interrupted append; complete checkpoint is authoritative
            raise
        if record[field] <= maximum:
            retained.append(line)
    path.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")


def required_checkpoint_space(config, method, model, start_step, stop_step):
    pending = [step for step in config["checkpoint_updates"] if start_step < step <= stop_step]
    if not pending:
        return 0
    if method in ("full", "l2sp"):
        model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        recovery_bytes = model_bytes * 3  # model plus FP32 Adam first and second moments
        inference_steps = {625, 1250} if method == "full" else {1250}
        inference_bytes = model_bytes * sum(step in inference_steps for step in pending)
    else:
        model_bytes = sum(t.numel() * t.element_size() for t in adapter_state(model).values())
        optimizer_bytes = trainable_parameters(model) * 8
        recovery_bytes = model_bytes + optimizer_bytes
        inference_bytes = model_bytes * (1250 in pending)
        if method == "oplora16" and 1250 in pending:
            inference_bytes += trainable_parameters(model) * 4
    return int(1.15 * (len(pending) * recovery_bytes + inference_bytes))


def single_recovery_bytes(method, model):
    if method in ("full", "l2sp"):
        return 3 * sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    return (sum(value.numel() * value.element_size() for value in adapter_state(model).values())
            + trainable_parameters(model) * 8)


def available_ram_bytes():
    if os.name != "posix":
        return None
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def require_checkpoint_ram(method, model):
    available = available_ram_bytes()
    if available is None:
        return None
    model_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    # A CPU model copy, recovery serialization/load, and 2 GiB OS headroom.
    needed = single_recovery_bytes(method, model) + model_bytes + 2 * 1024**3
    if available < needed:
        raise RuntimeError(f"{method} recovery needs about {needed/2**30:.1f} GiB available host RAM; "
                           f"only {available/2**30:.1f} GiB is available")
    return {"available_bytes": available, "required_bytes": needed}


def gpu_memory_snapshot():
    result = []
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        result.append({"device": device, "allocated_bytes": torch.cuda.memory_allocated(device),
                       "reserved_bytes": torch.cuda.memory_reserved(device),
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                       "free_bytes": free, "total_bytes": total})
    return result


def execute_update(config, method, model, optimizer, scheduler, scaler, reference,
                   batch, pad_id, update, skipped_path=None, skipped_before=0,
                   device="cuda:0"):
    """The one optimizer-update path used by training and real GPU preflight."""
    valid = sum(row["response_tokens"] for row in batch)
    if len(batch) != config["microbatch"] * config["grad_accum"] or valid <= 0:
        raise ValueError("Invalid effective batch")
    retries = 0
    while True:
        trial_rng = rng_state()
        optimizer.zero_grad(set_to_none=True)
        sft_sum = 0.0
        for offset in range(0, len(batch), config["microbatch"]):
            micro = collate(batch[offset:offset + config["microbatch"]], pad_id, device)
            with torch.autocast(device_type="cuda" if str(device).startswith("cuda") else "cpu",
                                dtype=torch.float16, enabled=str(device).startswith("cuda")):
                loss_sum = response_loss_sum(model, micro["input_ids"],
                                             micro["attention_mask"], micro["labels"])
                loss = loss_sum / valid
            sft_sum += loss_sum.detach().item()
            scaler.scale(loss).backward()
            del micro, loss_sum, loss
        scaler.unscale_(optimizer)
        omega = reference.omega_and_gradient(model, config["l2sp_lambda"]) if method == "l2sp" else 0.0
        norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                              config["max_grad_norm"]).item()
        if method in ("full", "l2sp"):
            probe_grad = model.model.layers[0].self_attn.q_proj.weight.grad
        else:
            probe_grad = model.model.layers[0].self_attn.q_proj.B.grad
        representative_grad_nonzero = (probe_grad is not None and
                                       torch.count_nonzero(probe_grad).item() > 0)
        if not math.isfinite(omega):
            raise FloatingPointError(f"Nonfinite L2-SP penalty at update {update}")
        old_scale = scaler.get_scale()
        used_lr = optimizer.param_groups[0]["lr"]
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            retries += 1
            if skipped_path is not None:
                with skipped_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"attempted_update": update, "retry": retries,
                                             "skipped_total": skipped_before + retries,
                                             "old_scale": old_scale, "new_scale": scaler.get_scale(),
                                             "grad_norm": norm if math.isfinite(norm) else str(norm)}) + "\n")
            if retries > 20:
                raise FloatingPointError(f"Repeated FP16 overflow at update {update}")
            restore_rng(trial_rng)
            continue
        if not math.isfinite(norm) or not math.isfinite(sft_sum):
            raise FloatingPointError(f"Nonfinite successful update {update}")
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        return {"sft_loss": sft_sum / valid, "omega": omega, "grad_norm": norm,
                "lr": used_lr, "skipped": retries,
                "representative_grad_nonzero": representative_grad_nonzero}


def run_training(config, method, stop_update=None):
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU")
    if config["train_examples"] != 40000 or config["microbatch"] != 2 or config["grad_accum"] != 16:
        raise ValueError("Assignment training budget changed")
    stop_update = stop_update or config["optimizer_updates"]
    if stop_update not in config["checkpoint_updates"]:
        raise ValueError("stop_update must be one of the assignment checkpoint updates")
    rows, manifest = load_selected(config)
    from task1.preflight import require_passed_preflight
    require_passed_preflight(config, manifest)
    run_dir = Path(config["output_root"]) / method
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / "latest.pt"
    resume = latest_path
    if not resume.exists():
        recoveries = sorted(run_dir.glob("recovery_*.pt"))
        if recoveries:
            resume = recoveries[-1]
        elif (run_dir / "train.jsonl").exists() or list(run_dir.glob("inference_*.pt")):
            raise RuntimeError(f"{method} has existing outputs but no recovery checkpoint; refusing to restart")
    set_seed(config["seed"])
    model, metadata = load_model(config, method, skip_basis=resume.exists() and method == "oplora16")
    model.train()
    tokenizer = AutoTokenizer.from_pretrained(config["model"], cache_dir=config["cache_dir"],
                                               revision=config.get("model_revision"))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    # Full FT also needs the complete W0 reference for its required displacement.
    reference = CPUReference(model, pin=method == "l2sp") if method in ("full", "l2sp") else None
    optimizer, optimizer_backend = build_optimizer(
        config, method, model, weight_decay_groups(model, config["weight_decay"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, config["warmup_updates"], config["optimizer_updates"])
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    step, skipped, elapsed = 0, 0, 0.0
    if resume.exists():
        expected = {"method": method, "data_sha256": manifest["selected_sha256"], "config": config}
        attempts = [resume] + [p for p in sorted(run_dir.glob("recovery_*.pt"), reverse=True) if p != resume]
        errors = []
        for candidate in attempts:
            try:
                state = load_checked(candidate, expected)
                resume = candidate
                break
            except (OSError, EOFError, RuntimeError, ValueError) as exc:
                errors.append(f"{candidate.name}: {exc}")
        else:
            raise RuntimeError("No valid recovery checkpoint: " + " | ".join(errors))
        if method in ("full", "l2sp"):
            model.load_state_dict(state["model"])
        else:
            load_adapter_state(model, state["model"])
        previous_backend = state.get("optimizer_backend", "single_gpu_adamw")
        migration = (method in ("full", "l2sp") and previous_backend == "single_gpu_adamw"
                     and optimizer_backend == "secondary_gpu_adamw")
        if previous_backend != optimizer_backend and not migration:
            raise ValueError("Recovery optimizer placement differs from the current GPU setup")
        if migration:
            print(f"Migrating {method} AdamW state from GPU 0 to GPU 1", flush=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        restore_rng(state["rng"])
        step, skipped, elapsed = state["step"], state["skipped"], state["elapsed"]
        metadata = state["metadata"]
        if state["position"] != step * 32:
            raise ValueError("Checkpoint data position does not match successful updates")
        print(f"Resuming {method} from update {step}, example {step*32}", flush=True)
        del state
        gc.collect()
    from common.checkpoint import atomic_json
    atomic_json(config, run_dir / "config.json", indent=2)
    (run_dir / "model_metadata.json").write_text(json.dumps({
        **metadata, "trainable_parameters": trainable_parameters(model)}, indent=2), encoding="utf-8")
    needed = required_checkpoint_space(config, method, model, step, stop_update)
    free = shutil.disk_usage(run_dir).free
    if free < needed:
        raise RuntimeError(f"Estimated {needed/2**30:.1f} GiB needed for pending recovery/inference files, "
                           f"but only {free/2**30:.1f} GiB free. Use --stop-update at the next "
                           "required checkpoint and persist each session's files before continuing.")
    if any(step < checkpoint <= stop_update for checkpoint in config["checkpoint_updates"]):
        require_checkpoint_ram(method, model)
    log_path = run_dir / "train.jsonl"
    if resume.exists() and log_path.exists():
        trim_jsonl_to_checkpoint(log_path, "update", step)
    skipped_path = run_dir / "skipped.jsonl"
    if resume.exists() and skipped_path.exists():
        trim_jsonl_to_checkpoint(skipped_path, "skipped_total", skipped)
    start = time.perf_counter()
    for device in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device)
    for target_step in range(step + 1, stop_update + 1):
        batch = rows[(target_step - 1) * 32:target_step * 32]
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device)
        update_result = execute_update(config, method, model, optimizer, scheduler, scaler,
                                       reference, batch, pad_id, target_step, skipped_path, skipped)
        skipped += update_result["skipped"]
        memory = gpu_memory_snapshot()
        step = target_step
        train_time = elapsed + time.perf_counter() - start
        penalty = config["l2sp_lambda"] * update_result["omega"] if method == "l2sp" else 0.0
        record = {"update": step, "examples": step * 32, "sft_loss": update_result["sft_loss"],
                  "omega": update_result["omega"], "penalty": penalty,
                  "total_loss": update_result["sft_loss"] + penalty,
                  "lr": update_result["lr"], "grad_norm": update_result["grad_norm"], "skipped_updates": skipped,
                  "training_seconds": train_time, "peak_gpu_bytes": memory[0]["peak_allocated_bytes"],
                  "gpu_memory": memory, "optimizer_backend": optimizer_backend,
                  "reference_bytes": reference.bytes if method == "l2sp" else 0,
                  "reference_transfer_seconds": reference.transfer_seconds if method == "l2sp" else 0}
        if method in ("full", "l2sp") and step in (625, 1250):
            displacement_omega = reference.omega_and_gradient(model)
            record["parameter_displacement_l2"] = math.sqrt(2 * displacement_omega)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if step in config["checkpoint_updates"]:
            require_checkpoint_ram(method, model)
            state = {"method": method, "data_sha256": manifest["selected_sha256"], "config": config,
                     "optimizer_backend": optimizer_backend,
                     "model": ({k: v.detach().cpu() for k, v in model.state_dict().items()}
                               if method in ("full", "l2sp") else adapter_state(model)),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "scaler": scaler.state_dict(), "rng": rng_state(), "step": step,
                     "position": step * 32, "skipped": skipped, "elapsed": train_time,
                     "metadata": metadata, "trainable_parameters": trainable_parameters(model)}
            recovery_path = run_dir / f"recovery_{step:04d}.pt"
            atomic_save(state, recovery_path)
            atomic_latest(recovery_path, latest_path)
            if (method == "full" and step in (625, 1250)) or (method == "l2sp" and step == 1250):
                atomic_save(state["model"], run_dir / f"inference_{step:04d}.pt")
            elif method not in ("full", "l2sp") and step == 1250:
                atomic_save(adapter_state(model), run_dir / f"inference_{step:04d}.pt")
                if method == "oplora16":
                    export_start = time.perf_counter()
                    projected_path = run_dir / f"projected_{step:04d}.pt"
                    atomic_save(projected_adapter_state(model), projected_path)
                    (run_dir / f"export_{step:04d}.json").write_text(json.dumps({
                        "seconds": time.perf_counter() - export_start,
                        "projected_bytes": projected_path.stat().st_size,
                        "basis_bytes": metadata["basis_bytes"],
                        "basis_seconds": metadata["basis_seconds"]}, indent=2), encoding="utf-8")
            del state
            gc.collect()
        if step % 10 == 0 or step in config["checkpoint_updates"]:
            print(f"{method}: {step}/1250, examples={step*32}, loss={record['sft_loss']:.4f}, "
                  f"peak={record['peak_gpu_bytes']/2**30:.2f} GiB", flush=True)
    return {"method": method, "updates": step, "skipped": skipped,
            "complete": step == config["optimizer_updates"]}
