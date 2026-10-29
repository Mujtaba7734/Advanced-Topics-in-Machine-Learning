"""Sustained, real-model memory and recovery checks for the five methods."""
import gc
import math
import shutil
import time
from pathlib import Path

import torch
from transformers import get_cosine_schedule_with_warmup

from common.checkpoint import atomic_save, load_checked
from common.seed import restore_rng, rng_state, set_seed
from task1.dataset import collate
from task1.l2sp import CPUReference
from task1.model import (AdapterLinear, adapter_state, load_adapter_state, load_model,
                         merge_adapters, projected_adapter_state, target_modules,
                         trainable_parameters, use_projected_factors)
from task1.optim import build_optimizer, optimizer_state_count
from task1.training import (execute_update, gpu_memory_snapshot,
                            required_checkpoint_space, require_checkpoint_ram, response_loss_sum,
                            weight_decay_groups)


SAFETY_BYTES = 1024 ** 3


def longest_consecutive_batches(rows, config):
    """Three 32-example updates from the longest prepared sequences, in stable order."""
    need = 3 * config["microbatch"] * config["grad_accum"]
    if len(rows) < need:
        raise ValueError(f"Sustained preflight needs {need} prepared examples")
    selected = sorted(rows, key=lambda row: len(row["input_ids"]), reverse=True)[:need]
    batch_size = config["microbatch"] * config["grad_accum"]
    return [selected[i:i + batch_size] for i in range(0, need, batch_size)]


def _assert_memory_margin(snapshots):
    for update, devices in enumerate(snapshots, 1):
        for device in devices:
            margin = min(device["total_bytes"] - device["peak_reserved_bytes"], device["free_bytes"])
            if margin < SAFETY_BYTES:
                raise RuntimeError(f"Update {update}, GPU {device['device']} peaked with only "
                                   f"{margin/2**30:.2f} GiB reserved-memory headroom; "
                                   "at least 1 GiB is required")


def _verify_structure(config, method, model):
    if not getattr(model, "is_gradient_checkpointing", False):
        raise AssertionError(f"{method} is missing gradient checkpointing")
    if method in ("full", "l2sp"):
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise AssertionError(f"{method} master parameters are not FP32")
        return
    expected_targets = len(model.model.layers) * len(config["targets"])
    modules = [module for module in model.modules() if isinstance(module, AdapterLinear)]
    if len(modules) != expected_targets:
        raise AssertionError(f"{method} has {len(modules)} adapters; expected {expected_targets}")
    expected_trainable = sum(module.rank * (module.base.in_features + module.base.out_features)
                             for module in modules)
    if trainable_parameters(model) != expected_trainable:
        raise AssertionError(f"{method} trainable parameter count differs from LoRA factor count")
    if any(parameter.requires_grad for module in modules for parameter in module.base.parameters()):
        raise AssertionError(f"{method} has unfrozen base weights")


def _merge_check(config, method, model, tokenizer):
    if method in ("full", "l2sp"):
        return
    model.eval()
    ids = tokenizer("Merge equivalence?", return_tensors="pt").input_ids.to("cuda:0")
    with torch.inference_mode():
        before = model(input_ids=ids).logits.float()
        if method == "oplora16":
            module = model.model.layers[0].mlp.down_proj
            if (module.uk.T @ module.correction()).abs().max().item() > 1e-3:
                raise AssertionError("OPLoRA protected output subspace is not orthogonal")
            projected = projected_adapter_state(model)
            use_projected_factors(model, projected)
            after_export = model(input_ids=ids).logits.float()
            if not torch.allclose(before, after_export, atol=0.1, rtol=0):
                raise AssertionError("OPLoRA projected export changed logits")
        merge_adapters(model)
        after_merge = model(input_ids=ids).logits.float()
        if not torch.allclose(before, after_merge, atol=0.1, rtol=0):
            raise AssertionError(f"{method} merge changed logits")


def inspect_model_method(config, method, rows, tokenizer, output_dir):
    gc.collect()
    torch.cuda.empty_cache()
    set_seed(config["seed"])
    model = optimizer = scheduler = scaler = reference = state = loaded = None
    path = Path(output_dir) / f"{method}_recovery_smoke.pt"
    try:
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device)
        model, metadata = load_model(config, method)
        setup_memory = gpu_memory_snapshot()
        _assert_memory_margin([setup_memory])
        model.train()
        _verify_structure(config, method, model)
        reference = (CPUReference(model, pin=method == "l2sp")
                     if method in ("full", "l2sp") else None)
        optimizer, backend = build_optimizer(
            config, method, model, weight_decay_groups(model, config["weight_decay"]))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, config["warmup_updates"], config["optimizer_updates"])
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        checkpoint_estimate = required_checkpoint_space(config, method, model, 0, 312)
        disk_free = shutil.disk_usage(config["output_root"]).free
        if disk_free < checkpoint_estimate:
            raise RuntimeError(f"{method} first recovery estimate is {checkpoint_estimate/2**30:.1f} GiB; "
                               f"disk has {disk_free/2**30:.1f} GiB free")
        batches = longest_consecutive_batches(rows, config)
        longest = max(len(row["input_ids"]) for batch in batches for row in batch)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        probe = (model.model.layers[0].self_attn.q_proj.weight if method in ("full", "l2sp")
                 else model.model.layers[0].self_attn.q_proj.B)
        original_probe = probe.detach().clone()
        trainable_count = trainable_parameters(model)
        frozen = (model.model.layers[0].self_attn.q_proj.base.weight.detach().clone()
                  if method not in ("full", "l2sp") else None)
        results, memory = [], [setup_memory]
        for update, batch in enumerate(batches, 1):
            for device in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            result = execute_update(config, method, model, optimizer, scheduler, scaler,
                                    reference, batch, pad_id, update)
            result["seconds"] = time.perf_counter() - started
            if not result["representative_grad_nonzero"]:
                raise AssertionError(f"{method} target gradient was zero at update {update}")
            results.append(result)
            memory.append(gpu_memory_snapshot())
        _assert_memory_margin(memory)
        if torch.equal(probe.detach(), original_probe):
            raise AssertionError(f"{method} representative trainable parameter did not update")
        if frozen is not None and not torch.equal(
                model.model.layers[0].self_attn.q_proj.base.weight, frozen):
            raise AssertionError(f"{method} changed a frozen backbone weight")
        if method in ("full", "l2sp") and reference.omega_and_gradient(model) <= 0:
            raise AssertionError(f"{method} displacement is zero after three updates")
        if method == "l2sp" and results[2]["omega"] <= 0:
            raise AssertionError("L2-SP penalty is zero on the third update")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise AssertionError("Training retained gradients after optimizer step")
        host_ram = require_checkpoint_ram(method, model)
        state = {"method": method, "step": 3, "position": 96,
                 "optimizer_backend": backend,
                 "model": ({key: value.detach().cpu() for key, value in model.state_dict().items()}
                           if method in ("full", "l2sp") else adapter_state(model)),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "rng": rng_state()}
        atomic_save(state, path)
        checkpoint_bytes = path.stat().st_size
        memory.append(gpu_memory_snapshot())
        _assert_memory_margin(memory)
        state = None
        gc.collect()
        loaded = load_checked(path, {"method": method, "step": 3, "position": 96,
                                     "optimizer_backend": backend})
        if optimizer_state_count(loaded["optimizer"]) == 0:
            raise AssertionError("Recovery checkpoint has no initialized Adam moments")
        trained_probe = probe.detach().clone()
        with torch.no_grad():
            probe.add_(0.01)
        if method in ("full", "l2sp"):
            model.load_state_dict(loaded["model"])
        else:
            load_adapter_state(model, loaded["model"])
        optimizer.load_state_dict(loaded["optimizer"])
        scheduler.load_state_dict(loaded["scheduler"])
        scaler.load_state_dict(loaded["scaler"])
        restore_rng(loaded["rng"])
        if not torch.equal(probe, trained_probe):
            raise AssertionError("Reload did not restore the trained parameter")
        loaded = None
        gc.collect()
        optimizer.zero_grad(set_to_none=True)
        micro = collate(batches[2][:2], pad_id, "cuda:0")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            continued_loss = response_loss_sum(model, micro["input_ids"],
                                               micro["attention_mask"], micro["labels"])
        if not torch.isfinite(continued_loss):
            raise FloatingPointError("Reloaded inference-mode forward is nonfinite")
        continued_value = continued_loss.item()
        del micro, continued_loss
        memory.append(gpu_memory_snapshot())
        _assert_memory_margin(memory)
        _merge_check(config, method, model, tokenizer)
        memory.append(gpu_memory_snapshot())
        _assert_memory_margin(memory)
        return {"optimizer_backend": backend, "sequence_max": longest,
                "updates": results, "gpu_memory": memory,
                "minimum_headroom_bytes": min(min(device["total_bytes"]-device["peak_reserved_bytes"],
                                                  device["free_bytes"])
                                              for devices in memory for device in devices),
                "checkpoint_estimate_bytes": checkpoint_estimate,
                "smoke_checkpoint_bytes": checkpoint_bytes,
                "reference_bytes": reference.bytes if reference else 0,
                "checkpoint_host_ram": host_ram,
                "worst_length_1250_update_projection_seconds":
                    sum(result["seconds"] for result in results) / len(results) * config["optimizer_updates"],
                "resume_forward_loss_sum": continued_value,
                "trainable_parameters": trainable_count, **metadata}
    finally:
        path.unlink(missing_ok=True)
        model = optimizer = scheduler = scaler = reference = state = loaded = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
