"""Restartable Task 1 stage runner used by the multi-cell Kaggle notebook.

Each invocation is a new Python process. Scientific libraries are imported only
after a clean subprocess has verified the active interpreter's dependencies.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path


METHODS = ("full", "l2sp", "lora4", "lora16", "oplora16")
CONDITIONS = ("baseline", "full_mid", "full_final", "l2sp_final",
              "lora4_final", "lora16_final", "oplora16_final")
TASKS = {"gsm8k_cot": 1319, "hellaswag": 10042, "boolq": 3270, "arc_easy": 2376}
STAGES = ("environment", "dependencies", "restore", "dataset", "preflight", "baseline",
          "train_full", "train_l2sp", "train_lora4", "train_lora16", "train_oplora16",
          "evaluate_models", "weight", "activations", "qualitative", "aggregate",
          "final_verify", "export")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def source_revision(repo):
    manifest = read_json(Path(repo) / "kaggle" / "source_manifest.json")
    if not isinstance(manifest, dict) or not manifest.get("files"):
        raise RuntimeError("Audited source_manifest.json is missing")
    calculated = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True,
                                   separators=(",", ":")).encode()).hexdigest()
    if manifest.get("revision") != calculated:
        raise RuntimeError("Audited source revision does not match its file hashes")
    for relative, expected in manifest["files"].items():
        path = Path(repo) / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Audited source differs at {path}; mount the current Assignment-3 folder")
    return manifest["revision"]


def package_requirements(repo):
    pins = {}
    for line in (Path(repo) / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            name, version = line.strip().split("==", 1)
            pins[name] = version
    return pins


def mismatched_requirements(repo):
    bad = []
    for name, required in package_requirements(repo).items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            bad.append(f"{name}=={required}")
            continue
        if actual.split("+")[0] != required.split("+")[0]:
            bad.append(f"{name}=={required}")
    return bad


IMPORT_PROBE = r'''
import json
import torch
import transformers
from transformers import Qwen3ForCausalLM, AutoTokenizer
import datasets, accelerate, peft
import lm_eval
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
import numpy, pandas, scipy, matplotlib, yaml
import safetensors, packaging
print(json.dumps({"python": __import__("sys").executable,
                  "torch": torch.__version__, "cuda": torch.cuda.is_available(),
                  "gpus": torch.cuda.device_count()}))
'''


def import_probe():
    return subprocess.run([sys.executable, "-c", IMPORT_PROBE], text=True,
                          capture_output=True, timeout=180)


def _pip(*arguments):
    subprocess.run([sys.executable, "-m", "pip", *arguments], check=True,
                   env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"})


def ensure_dependencies(repo):
    """Check metadata and real imports in the same interpreter as every stage."""
    bad = mismatched_requirements(repo)
    if bad:
        print(f"[RUNNING] Dependencies  -  installing {', '.join(bad)}", flush=True)
        _pip("install", "--quiet", *bad)
    probe = import_probe()
    if probe.returncode != 0 and "torchvision::nms" in probe.stderr:
        try:
            importlib.metadata.version("torchvision")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            print("[RUNNING] Dependencies  -  removing broken optional TorchVision", flush=True)
            _pip("uninstall", "--yes", "--quiet", "torchvision")
            probe = import_probe()
    if probe.returncode != 0:
        missing = re.search(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)", probe.stderr)
        if missing:
            module = missing.group(1).split(".")[0]
            package = {"lm_eval": "lm-eval", "yaml": "PyYAML"}.get(module, module)
            pin = package_requirements(repo).get(package)
            if pin:
                print(f"[RUNNING] Dependencies - repairing broken {package} import", flush=True)
                _pip("install", "--quiet", "--force-reinstall", "--no-deps", f"{package}=={pin}")
            else:
                print(f"[RUNNING] Dependencies - repairing missing transitive module {module}", flush=True)
                _pip("install", "--quiet", "-r", str(Path(repo) / "requirements.txt"))
            probe = import_probe()
    if probe.returncode != 0:
        raise RuntimeError("Clean import probe failed:\n" + probe.stderr[-5000:])
    remaining = mismatched_requirements(repo)
    if remaining:
        raise RuntimeError(f"Pinned dependency versions remain incompatible: {remaining}")
    identity = json.loads(probe.stdout.strip().splitlines()[-1])
    if Path(identity["python"]).resolve() != Path(sys.executable).resolve():
        raise RuntimeError("Dependency probe used a different Python interpreter")
    return identity


class Blocked(Exception):
    """A safe, recoverable stop such as waiting for a mounted durable backup."""


class Context:
    def __init__(self, repo, work):
        self.repo = Path(repo).resolve()
        self.work = Path(work).resolve()
        self.output = self.work / "outputs" / "task1"
        self.results = self.work / "results" / "task1"
        self.state_path = self.output / "stage_state.json"
        self.revision = source_revision(self.repo)
        self.output.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)

    def state(self):
        return read_json(self.state_path, {"stages": {}, "source_revision": self.revision})

    def record(self, stage, status, **details):
        state = self.state()
        state.setdefault("stages", {})[stage] = {"status": status, "utc": time.time(), **details}
        state["source_revision"] = self.revision
        atomic_json(self.state_path, state)

    def config(self):
        import yaml
        from task1.protocol import config_sha256, validate_protocol
        config = yaml.safe_load((self.repo / "configs" / "task1.yaml").read_text(encoding="utf-8"))
        config["output_root"] = str(self.output)
        config["result_root"] = str(self.results)
        config["cache_dir"] = str(self.work / "hf_cache")
        prompt = self.repo / "first_plot_questions.yaml"
        if not prompt.is_file():
            matches = list(self.work.glob("first_plot_questions.yaml"))
            matches += list((self.work / "inputs").glob("*/first_plot_questions.yaml"))
            if len(matches) != 1:
                raise FileNotFoundError("Mount the official first_plot_questions.yaml with the source")
            prompt = matches[0]
        stable_prompt = self.work / "first_plot_questions.yaml"
        if prompt != stable_prompt:
            if stable_prompt.is_file() and sha256_file(stable_prompt) != sha256_file(prompt):
                raise RuntimeError("Working prompt YAML differs from the mounted assignment YAML")
            if not stable_prompt.exists():
                shutil.copy2(prompt, stable_prompt)
        manifest = read_json(self.output / "data" / "manifest.json")
        revision = manifest.get("model_revision") if manifest else None
        marker = read_json(self.output / "data" / "preflight_pass.json", {})
        # Preserve a valid prior preflight whose only path/revision choices came
        # from the older notebook. Otherwise use a stable working prompt copy.
        candidates = [(stable_prompt, revision), (prompt, revision),
                      (prompt, config.get("model_revision")),
                      (stable_prompt, config.get("model_revision"))]
        original = dict(config)
        for prompt_path, model_revision in candidates:
            candidate = {**original, "qualitative_prompts": str(prompt_path),
                         "model_revision": model_revision}
            if marker.get("config_sha256") == config_sha256(candidate):
                config = candidate
                break
        else:
            config = {**original, "qualitative_prompts": str(stable_prompt),
                      "model_revision": revision or original.get("model_revision")}
        validate_protocol(config)
        runtime = self.work / "task1_runtime.yaml"
        rendered = yaml.safe_dump(config, sort_keys=False)
        if not runtime.is_file() or runtime.read_text(encoding="utf-8") != rendered:
            runtime.write_text(rendered, encoding="utf-8")
        os.environ["HF_HOME"] = config["cache_dir"]
        os.environ["HF_DATASETS_CACHE"] = str(Path(config["cache_dir"]) / "datasets")
        return config

    def output_mounts(self):
        input_root = Path("/kaggle/input")
        if not input_root.is_dir():
            return []
        return [path for path in input_root.iterdir() if (path / "outputs" / "task1").is_dir()]


def stage_environment(ctx):
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True)
        devices = gpu.stdout.strip().splitlines() if gpu.returncode == 0 else []
    except FileNotFoundError:
        devices = []
    ram = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
           if os.name == "posix" else None)
    disk = shutil.disk_usage(ctx.work)
    details = {"python": sys.version.split()[0], "interpreter": sys.executable,
               "kaggle": Path("/kaggle/working").is_dir(), "gpus": devices,
               "ram_gib": round(ram / 2**30, 2) if ram else None,
               "disk_free_gib": round(disk.free / 2**30, 2),
               "source_revision": ctx.revision}
    print("[PASS] Environment  -  " + json.dumps(details, ensure_ascii=False), flush=True)
    return details


def _prepared_pair_valid(data_dir):
    selected, manifest = data_dir / "selected.json", data_dir / "manifest.json"
    if not selected.is_file() or not manifest.is_file():
        return False
    try:
        info = read_json(manifest)
        if sha256_file(selected) != info["selected_sha256"] or len(info["indices"]) != 40000:
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _copy_unique_sources(sources, target):
    if target.exists():
        if any(sha256_file(source) != sha256_file(target) for source in sources):
            print(f"[SKIPPED] Preserved working copy that differs from mounted copy: {target}", flush=True)
        return False
    hashes = {sha256_file(source) for source in sources}
    if len(hashes) != 1:
        raise Blocked(f"Ambiguous mounted versions for {target}; attach one current output dataset")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".restore.tmp")
    shutil.copy2(sources[0], temporary)
    if sha256_file(temporary) not in hashes:
        temporary.unlink(missing_ok=True)
        raise IOError(f"Restored file checksum changed: {target}")
    os.replace(temporary, target)
    return True


def stage_restore(ctx):
    mounts = ctx.output_mounts()
    if not mounts:
        print("[PASS] Recovery  -  no mounted output dataset; keeping working files", flush=True)
        return {"mounts": 0, "copied": 0}
    # Environment/dependency cells already create a current-session state file.
    # Recover prior stage records and pending backup before copying other files.
    current = ctx.state()
    saved_states = [read_json(mount / "outputs/task1/stage_state.json") for mount in mounts
                    if (mount / "outputs/task1/stage_state.json").is_file()]
    pending_values = [state.get("backup_pending") for state in saved_states
                      if state.get("backup_pending")]
    if len({json.dumps(value, sort_keys=True) for value in pending_values}) > 1:
        raise Blocked("Mounted output datasets have conflicting pending checkpoint states")
    for saved in saved_states:
        for name, record in saved.get("stages", {}).items():
            current.setdefault("stages", {}).setdefault(name, record)
    if pending_values and not current.get("backup_pending"):
        current["backup_pending"] = pending_values[0]
    atomic_json(ctx.state_path, current)
    data_dir = ctx.output / "data"
    if not _prepared_pair_valid(data_dir):
        candidates = [mount / "outputs" / "task1" / "data" for mount in mounts]
        valid = [source for source in candidates if _prepared_pair_valid(source)]
        selected_hashes = {read_json(source / "manifest.json")["selected_sha256"] for source in valid}
        if len(selected_hashes) > 1:
            raise Blocked("Mounted output datasets contain different valid training selections")
        if valid:
            data_dir.mkdir(parents=True, exist_ok=True)
            for name in ("selected.json", "manifest.json"):
                target = data_dir / name
                if target.exists():
                    quarantine = data_dir / f"{name}.invalid.{int(time.time())}"
                    target.rename(quarantine)
                    print(f"[RECOVERED] Quarantined invalid {target.name} as {quarantine.name}", flush=True)
                shutil.copy2(valid[0] / name, target)
            if not _prepared_pair_valid(data_dir):
                raise IOError("Restored prepared data failed integrity verification")
    selected = {}
    for mount in mounts:
        for relative_root in (Path("outputs/task1"), Path("results/task1")):
            source_root = mount / relative_root
            if not source_root.is_dir():
                continue
            for source in source_root.rglob("*"):
                if not source.is_file():
                    continue
                relative = source.relative_to(mount)
                if relative.name == "stage_state.json":
                    continue
                if relative.name in ("selected.json", "manifest.json") and "data" in relative.parts:
                    continue
                if relative.name.startswith("recovery_") and relative.suffix == ".pt":
                    siblings = sorted(source.parent.glob("recovery_*.pt"))
                    if source != siblings[-1]:
                        continue
                selected.setdefault(relative, []).append(source)
    copied = 0
    for relative, sources in sorted(selected.items(), key=lambda pair: str(pair[0])):
        if relative.name == "latest.pt" and any(
                candidate.parent == relative.parent and candidate.name.startswith("recovery_")
                for candidate in selected):
            continue  # latest and newest recovery are the same logical state.
        target = ctx.work / relative
        copied += int(_copy_unique_sources(sources, target))
    state = ctx.state()
    pending = state.get("backup_pending")
    if pending:
        wanted = pending["sha256"]
        method = pending["method"]
        found = any(sha256_file(source) == wanted for mount in mounts
                    for source in (mount / "outputs" / "task1" / method).glob("recovery_*.pt"))
        if found:
            state.pop("backup_pending")
            atomic_json(ctx.state_path, state)
            print(f"[PASS] Durable mounted recovery verified for {method}", flush=True)
    print(f"[PASS] Recovery  -  {len(mounts)} mounted dataset(s), {copied} files restored", flush=True)
    return {"mounts": len(mounts), "copied": copied}


def _data_status(config):
    from task1.dataset import load_selected
    rows, manifest = load_selected(config)
    probe_path = Path(config["output_root"]) / "data" / "probe_ids.json"
    probes = read_json(probe_path)
    if not probes or any(
            len(probes.get(name, [])) != 32 or
            probes[name] != sorted(set(probes[name])) or
            any(not isinstance(value, int) or value < 0 or value >= upper
                for value in probes[name])
            for name, upper in (("gsm8k", 1319), ("hellaswag", 10042))):
        raise ValueError("Prepared data has missing or invalid 32+32 probe IDs")
    return {"rows": len(rows), "data_sha256": manifest["selected_sha256"],
            "probe_counts": {name: len(probes[name]) for name in ("gsm8k", "hellaswag")}}


def stage_dataset(ctx, config):
    try:
        status = _data_status(config)
    except FileNotFoundError:
        status = None
    except ValueError as exc:
        data_dir = ctx.output / "data"
        if (_prepared_pair_valid(data_dir) and
                not (data_dir / "probe_ids.json").exists()):
            from task1.diagnostics import create_probe_ids
            create_probe_ids(config)
            status = _data_status(config)
        else:
            raise Blocked(f"Prepared data validation failed: {exc}") from exc
    if status:
        print("DATASET: VERIFIED  -  SKIPPING PREPARATION", flush=True)
        return status
    data = ctx.output / "data"
    if (data / "selected.json").exists() or (data / "manifest.json").exists():
        if not _prepared_pair_valid(data):
            raise Blocked("Prepared dataset is incomplete or corrupt; mount a valid saved dataset")
    from task1.prepare_data import prepare
    from task1.diagnostics import create_probe_ids
    manifest = prepare(config)
    create_probe_ids(config)
    status = _data_status(config)
    print(f"[PASS] Dataset  -  {status['rows']:,} verified, SHA {status['data_sha256'][:12]}", flush=True)
    return status


def stage_preflight(ctx, config):
    from task1.dataset import load_selected
    from task1.preflight import require_passed_preflight, run_preflight
    _, manifest = load_selected(config)
    try:
        require_passed_preflight(config, manifest)
    except (RuntimeError, FileNotFoundError, KeyError, ValueError):
        pass
    else:
        print("[SKIPPED] GPU preflight  -  matching pass report, data, code, and hardware", flush=True)
        return {"status": "verified_existing"}
    report = run_preflight(config, ctx.repo / "requirements.txt")
    require_passed_preflight(config, manifest)
    print("[PASS] GPU preflight  -  " + ", ".join(report["checks"]), flush=True)
    return {"report": str(ctx.output / "preflight" / "report.json")}


def _condition_status(config, condition):
    from task1.evaluation import condition_fingerprint
    from task1.plots import PRIMARY
    fingerprint = condition_fingerprint(config, condition)
    root = Path(config["result_root"]) / "evaluation" / condition
    complete = []
    for task, expected in TASKS.items():
        path = root / f"{task}.json"
        if not path.is_file():
            continue
        record = read_json(path)
        if (record.get("condition") != condition or record.get("task") != task or
                record.get("condition_fingerprint") != fingerprint):
            raise Blocked(f"Stale or incomplete evaluation file: {path}")
        if record.get("sample_count") != expected:
            continue  # evaluate() recomputes incomplete task files atomically.
        if (len(record.get("samples", [])) != expected or
                PRIMARY[task] not in record.get("results", {})):
            raise Blocked(f"Corrupt completed evaluation file: {path}")
        complete.append(task)
    return complete


def bounded_evaluation_smoke(config, condition, work):
    """Call the real evaluate() entry point with only two records per task."""
    import lm_eval
    import task1.evaluation as evaluation
    original_simple = lm_eval.simple_evaluate
    original_tasks = evaluation.TASKS
    with tempfile.TemporaryDirectory(prefix="eval_smoke_", dir=work) as temp:
        smoke_config = {**config, "result_root": str(Path(temp) / "results")}

        def bounded(*args, **kwargs):
            kwargs["limit"] = 2
            if task == "gsm8k_cot":
                kwargs["gen_kwargs"] = {"max_gen_toks": 64, "do_sample": False}
            return original_simple(*args, **kwargs)

        lm_eval.simple_evaluate = bounded
        evaluation.TASKS = {task: 2 for task in TASKS}
        try:
            result = evaluation.evaluate(smoke_config, condition)
            if set(result) != set(TASKS):
                raise AssertionError("Bounded evaluation did not exercise all four tasks")
        finally:
            evaluation.TASKS = original_tasks
            lm_eval.simple_evaluate = original_simple
    print("[PASS] Bounded evaluation smoke  -  actual evaluate() path, no official output", flush=True)


def stage_baseline(ctx, config):
    complete = _condition_status(config, "baseline")
    if len(complete) == len(TASKS):
        print("[SKIPPED] Baseline  -  four verified full-split results", flush=True)
        return {"tasks": complete, "status": "verified_existing"}
    bounded_evaluation_smoke(config, "baseline", ctx.work)
    from task1.evaluation import evaluate
    evaluate(config, "baseline")
    complete = _condition_status(config, "baseline")
    if len(complete) != len(TASKS):
        raise RuntimeError("Baseline evaluation did not save all four complete tasks")
    print("[PASS] Baseline  -  four complete benchmark splits", flush=True)
    return {"tasks": complete}


def stage_evaluate_models(ctx, config):
    from task1.evaluation import evaluate
    done = {}
    for condition in CONDITIONS[1:]:
        complete = _condition_status(config, condition)
        if len(complete) == len(TASKS):
            print(f"[SKIPPED] {condition}  -  four verified results", flush=True)
        else:
            evaluate(config, condition)
            complete = _condition_status(config, condition)
            if len(complete) != len(TASKS):
                raise RuntimeError(f"{condition} evaluation remains incomplete")
            print(f"[PASS] {condition}  -  four complete splits", flush=True)
        done[condition] = complete
    return done


def _checkpoint_progress(config, method):
    import torch
    root = Path(config["output_root"]) / method
    candidates = sorted(root.glob("recovery_*.pt"), reverse=True)
    if (root / "latest.pt").is_file():
        candidates.append(root / "latest.pt")
    if not candidates:
        if (root / "train.jsonl").exists() or list(root.glob("inference_*.pt")):
            raise Blocked(f"{method} has outputs but no recovery checkpoint")
        return 0, None
    from task1.dataset import load_selected
    _, manifest = load_selected(config)
    errors = []
    for path in candidates:
        try:
            state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            step = state["step"]
            if (state.get("method") != method or state.get("data_sha256") != manifest["selected_sha256"] or
                    state.get("config") != config or state.get("position") != step * 32 or
                    step not in config["checkpoint_updates"]):
                raise ValueError("checkpoint identity or position mismatch")
            del state
            return step, path
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{path.name}: {exc}")
    raise Blocked(f"{method} has no valid recovery checkpoint: {' | '.join(errors)}")


def _last_logged_update(path):
    if not path.is_file():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        try:
            return int(json.loads(line)["update"])
        except (ValueError, KeyError, TypeError):
            continue
    return 0


def _method_complete(config, method, step):
    if step != config["optimizer_updates"]:
        return False
    root = Path(config["output_root"]) / method
    required = [root / "inference_1250.pt"]
    if method == "full":
        required.append(root / "inference_0625.pt")
    if method == "oplora16":
        required.append(root / "projected_1250.pt")
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        return False
    return _last_logged_update(root / "train.jsonl") >= 1250


def _checkpoint_size_from_preflight(ctx, method):
    report = read_json(ctx.output / "preflight" / "report.json")
    name = f"real_model_optimizer_checkpoint_{method}"
    detail = (report or {}).get("checks", {}).get(name, {}).get("detail", {})
    size = detail.get("smoke_checkpoint_bytes")
    if not isinstance(size, int) or size <= 0:
        raise Blocked(f"No measured recovery size for {method}; run the matching GPU preflight")
    return size, detail


def _synchronize_latest(checkpoint, method_root):
    """Make the trainer resume the verified newest recovery, not an older alias."""
    latest = Path(method_root) / "latest.pt"
    checkpoint = Path(checkpoint)
    if checkpoint == latest:
        return
    if not latest.is_file() or not os.path.samefile(checkpoint, latest):
        from common.checkpoint import atomic_latest
        atomic_latest(checkpoint, latest)


def choose_stop_update(config, method, step, free_bytes, recovery_bytes,
                       seconds_per_update=None, seconds_remaining=None):
    """Choose the furthest official checkpoint that fits disk and session budget."""
    reserve = 2 * 1024**3
    chosen = None
    for candidate in config["checkpoint_updates"]:
        if candidate <= step:
            continue
        pending = [value for value in config["checkpoint_updates"] if step < value <= candidate]
        model_bytes = recovery_bytes / 3 if method in ("full", "l2sp") else recovery_bytes
        inference_count = (sum(value in (625, 1250) for value in pending) if method == "full"
                           else 2 * int(1250 in pending) if method == "oplora16"
                           else int(1250 in pending))
        estimated_disk = int(1.15 * (len(pending) * recovery_bytes + inference_count * model_bytes))
        if estimated_disk + reserve > free_bytes:
            break
        if (seconds_per_update and seconds_remaining is not None and
                seconds_per_update * (candidate - step) > seconds_remaining):
            break
        chosen = candidate
    return chosen


def _session_seconds_remaining(ctx):
    state = ctx.state()
    session = state.get("session", {})
    parent_pid = os.getppid()
    parent_proc = Path(f"/proc/{parent_pid}")
    parent_identity = (f"{parent_pid}:{parent_proc.stat().st_ctime_ns}"
                       if parent_proc.exists() else str(parent_pid))
    if session.get("parent_identity") != parent_identity:
        session = {"parent_identity": parent_identity, "started_utc": time.time()}
        state["session"] = session
        atomic_json(ctx.state_path, state)
    # Conservative budget; actual Kaggle limits can be shorter.
    return max(0, 8 * 3600 - (time.time() - session["started_utc"]))


def stage_train(ctx, config, method):
    pending = ctx.state().get("backup_pending")
    if pending:
        raise Blocked(f"Mount a durable output dataset containing {pending['method']} checkpoint "
                      f"{pending['step']} before continuing training")
    step, checkpoint = _checkpoint_progress(config, method)
    if _method_complete(config, method, step):
        print(f"[SKIPPED] {method}  -  verified 1,250 updates", flush=True)
        return {"step": 1250, "status": "verified_existing"}
    recovery_size, detail = _checkpoint_size_from_preflight(ctx, method)
    free = shutil.disk_usage(ctx.output).free
    projected = detail.get("worst_length_1250_update_projection_seconds")
    seconds_per_update = projected / 1250 if isinstance(projected, (float, int)) and projected > 0 else None
    stop = choose_stop_update(config, method, step, free, recovery_size,
                              seconds_per_update, _session_seconds_remaining(ctx))
    if stop is None:
        raise Blocked(f"{method} cannot reach its next official checkpoint with safe disk/time margin; "
                      "save the current output dataset and use a fresh session or larger hardware")
    if checkpoint:
        _synchronize_latest(checkpoint, Path(config["output_root"]) / method)
    print(f"[{'RESUMED' if step else 'RUNNING'}] {method}  -  update {step}/1250 to {stop}/1250", flush=True)
    from task1.training import run_training
    result = run_training(config, method, stop_update=stop)
    verified_step, verified_checkpoint = _checkpoint_progress(config, method)
    if verified_step != stop or not verified_checkpoint:
        raise RuntimeError(f"{method} did not save the planned recovery checkpoint {stop}")
    if result["complete"] and _method_complete(config, method, stop):
        print(f"[PASS] {method}  -  complete at 1,250 updates", flush=True)
        return {"step": stop, "complete": True}
    digest = sha256_file(verified_checkpoint)
    state = ctx.state()
    state["backup_pending"] = {"method": method, "step": stop, "sha256": digest,
                               "checkpoint": str(verified_checkpoint)}
    atomic_json(ctx.state_path, state)
    print(f"[BLOCKED] {method} paused at verified checkpoint {stop}; "
          "use Kaggle Save Version, mount that output dataset, then Run All again", flush=True)
    return {"step": stop, "complete": False, "backup_pending": True}


def _recorded_artifacts_valid(ctx, stage):
    record = ctx.state().get("stages", {}).get(stage, {})
    if record.get("status") != "PASS" or record.get("source_revision") != ctx.revision:
        return False
    for relative, expected in record.get("artifacts", {}).items():
        path = ctx.work / relative
        if not path.is_file() or sha256_file(path) != expected:
            return False
    return bool(record.get("artifacts"))


def stage_weight(ctx, config):
    target = ctx.results / "diagnostics" / "weight_space.csv"
    if _recorded_artifacts_valid(ctx, "weight"):
        print("[SKIPPED] Weight diagnostics  -  verified output", flush=True)
        return {"artifacts": [target]}
    from task1.diagnostics import run_weight_diagnostics
    rows = run_weight_diagnostics(config)
    if len(rows) != 18 or not target.is_file():
        raise RuntimeError("Expected six conditions x three weight diagnostic layers")
    print("[PASS] Weight diagnostics  -  18 condition/layer records", flush=True)
    return {"artifacts": [target]}


def stage_activations(ctx, config):
    target = ctx.results / "diagnostics" / "activation_ratios.csv"
    if _recorded_artifacts_valid(ctx, "activations"):
        print("[SKIPPED] Activation diagnostics  -  verified output", flush=True)
        return {"artifacts": [target]}
    from task1.diagnostics import run_activation_diagnostics
    rows = run_activation_diagnostics(config)
    if len(rows) != 6 * 3 * 2 * 32 or not target.is_file():
        raise RuntimeError("Expected 1,152 activation diagnostic records")
    print("[PASS] Activation diagnostics  -  1,152 records", flush=True)
    return {"artifacts": [target, ctx.output / "data" / "baseline_activations.pt"]}


def _qualitative_status(config):
    from task1.evaluation import condition_fingerprint
    from task1.qualitative import load_prompts, prompt_sha256, validate_saved_rows
    rows, _ = load_prompts(config["qualitative_prompts"])
    fingerprint = prompt_sha256(config["qualitative_prompts"])
    root = Path(config["result_root"]) / "qualitative"
    complete = []
    for condition in CONDITIONS:
        path = root / f"{condition}.json"
        if not path.is_file():
            continue
        values = read_json(path)
        if len(values) < 24:
            continue  # generate() validates the saved prefix and resumes it.
        validate_saved_rows(values, rows, fingerprint)
        model_hash = condition_fingerprint(config, condition)
        if any(value.get("condition") != condition or
               value.get("condition_fingerprint") != model_hash or
               "response" not in value for value in values):
            raise Blocked(f"Qualitative output has stale model or missing response: {path}")
        complete.append(condition)
    return complete


def stage_qualitative(ctx, config):
    from task1.qualitative import annotation_template, combine_outputs, generate
    complete = _qualitative_status(config)
    for condition in CONDITIONS:
        if condition in complete:
            print(f"[SKIPPED] Qualitative {condition}  -  24 verified responses", flush=True)
            continue
        values = generate(config, condition)
        if len(values) != 24:
            raise RuntimeError(f"Qualitative {condition} produced {len(values)} responses")
        print(f"[PASS] Qualitative {condition}  -  24 responses", flush=True)
    if len(_qualitative_status(config)) != 7:
        raise RuntimeError("Qualitative generation is not complete for seven conditions")
    combined = combine_outputs(config)
    annotation = annotation_template(config)
    import pandas as pd
    if len(pd.read_csv(combined)) != 168 or len(pd.read_csv(annotation)) != 42:
        raise RuntimeError("Qualitative 168-response or 42-row annotation cardinality mismatch")
    return {"artifacts": [combined, annotation] +
            [ctx.results / "qualitative" / f"{condition}.json" for condition in CONDITIONS]}


def stage_aggregate(ctx, config):
    for condition in CONDITIONS:
        if len(_condition_status(config, condition)) != len(TASKS):
            raise Blocked(f"Cannot aggregate before {condition} has all four full evaluations")
    if len(_qualitative_status(config)) != 7:
        raise Blocked("Cannot aggregate before all 168 qualitative responses exist")
    if _recorded_artifacts_valid(ctx, "aggregate"):
        print("[SKIPPED] Results aggregation  -  verified outputs", flush=True)
        return {"artifacts": [ctx.results / "summary" / "benchmarks.csv"]}
    from task1.plots import aggregate
    from task1.qualitative import annotation_template, combine_outputs
    summary = aggregate(config)
    combined = combine_outputs(config)
    annotation = annotation_template(config)
    required = [summary / name for name in ("benchmarks.csv", "benchmarks_wide.csv",
                 "benchmarks.png", "sft_loss.png", "l2sp.png", "resources.csv",
                 "weight_space.png", "activation_ratios.png")]
    required += [combined, annotation]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Aggregation omitted required outputs: {missing}")
    print("[PASS] Aggregation  -  benchmark, training, resource, diagnostic, qualitative outputs", flush=True)
    return {"artifacts": required}


def final_requirements(ctx, config):
    """Machine-checkable computational checklist; the student PDF report is separate."""
    status = {}
    pdf = ctx.repo / "ATML-PA3.pdf"
    traceability = ctx.repo / "docs" / "task1_traceability.md"
    status["assignment_pdf"] = "PASS" if pdf.is_file() else "FAILED"
    status["traceability"] = "PASS" if traceability.is_file() else "FAILED"
    try:
        data = _data_status(config)
        status["40k_selected_and_64_probes"] = "PASS" if data["rows"] == 40000 else "FAILED"
    except (OSError, ValueError, KeyError):
        status["40k_selected_and_64_probes"] = "INCOMPLETE"
    from task1.dataset import load_selected
    from task1.preflight import require_passed_preflight
    try:
        _, manifest = load_selected(config)
        require_passed_preflight(config, manifest)
        status["real_gpu_preflight"] = "PASS"
    except (OSError, RuntimeError, ValueError, KeyError):
        status["real_gpu_preflight"] = "INCOMPLETE"
    for method in METHODS:
        try:
            step, _ = _checkpoint_progress(config, method)
            status[f"training_{method}"] = "PASS" if _method_complete(config, method, step) else "INCOMPLETE"
        except (Blocked, OSError, ValueError):
            status[f"training_{method}"] = "INCOMPLETE"
    for condition in CONDITIONS:
        try:
            complete = _condition_status(config, condition)
            for task in TASKS:
                status[f"evaluation_{condition}_{task}"] = "PASS" if task in complete else "INCOMPLETE"
        except (Blocked, OSError, ValueError):
            for task in TASKS:
                status[f"evaluation_{condition}_{task}"] = "INCOMPLETE"
    for name, path in (("weight_diagnostics", ctx.results / "diagnostics" / "weight_space.csv"),
                       ("activation_diagnostics", ctx.results / "diagnostics" / "activation_ratios.csv"),
                       ("benchmark_table", ctx.results / "summary" / "benchmarks.csv"),
                       ("resource_summary", ctx.results / "summary" / "resources.csv"),
                       ("training_figure", ctx.results / "summary" / "sft_loss.png"),
                       ("weight_figure", ctx.results / "summary" / "weight_space.png"),
                       ("activation_figure", ctx.results / "summary" / "activation_ratios.png")):
        status[name] = "PASS" if path.is_file() and path.stat().st_size else "INCOMPLETE"
    try:
        status["168_qualitative_responses"] = "PASS" if len(_qualitative_status(config)) == 7 else "INCOMPLETE"
    except (Blocked, OSError, ValueError):
        status["168_qualitative_responses"] = "INCOMPLETE"
    annotation = ctx.results / "qualitative" / "annotation_template.csv"
    status["42_annotation_template"] = "PASS" if annotation.is_file() else "INCOMPLETE"
    bundle = ctx.work / "task1_results_bundle.zip"
    index = read_json(ctx.work / "task1_results_bundle.index.json", {})
    if not bundle.is_file():
        status["compact_export"] = "NOT RUN"
    elif index.get("sha256") != sha256_file(bundle):
        status["compact_export"] = "FAILED"
    else:
        try:
            with zipfile.ZipFile(bundle) as archive:
                status["compact_export"] = "PASS" if archive.testzip() is None else "FAILED"
        except (OSError, zipfile.BadZipFile):
            status["compact_export"] = "FAILED"
    status["student_written_report"] = "NOT RUN"
    return status


def stage_final_verify(ctx, config):
    status = final_requirements(ctx, config)
    report = {"status": status, "source_revision": ctx.revision,
              "assignment_pdf_sha256": sha256_file(ctx.repo / "ATML-PA3.pdf")
              if (ctx.repo / "ATML-PA3.pdf").is_file() else None,
              "traceability_sha256": sha256_file(ctx.repo / "docs/task1_traceability.md")
              if (ctx.repo / "docs/task1_traceability.md").is_file() else None,
              "computational_complete": all(value == "PASS" for key, value in status.items()
                                            if key not in ("compact_export", "student_written_report")),
              "paths": {"outputs": str(ctx.output), "results": str(ctx.results)}}
    path = ctx.results / "summary" / "final_verification.json"
    atomic_json(path, report)
    counts = {value: list(status.values()).count(value) for value in ("PASS", "FAILED", "INCOMPLETE", "NOT RUN")}
    print("[PASS] Final verification report  -  " + json.dumps(counts) +
          f"; computational_complete={report['computational_complete']}", flush=True)
    return {"artifacts": [path], "computational_complete": report["computational_complete"]}


def stage_export(ctx, config):
    """Bundle small outputs; leave full recovery files for Kaggle Save Version."""
    roots = (ctx.output, ctx.results)
    skip_names = {"backup_manifest.json", "stage_state.json"}
    entries = []
    included = []
    inode_hashes = {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name in skip_names or path.name.endswith(".tmp"):
                continue
            info = path.stat()
            inode = (info.st_dev, info.st_ino, info.st_size)
            digest = inode_hashes.get(inode)
            if digest is None:
                digest = sha256_file(path)
                inode_hashes[inode] = digest
            relative = path.relative_to(ctx.work).as_posix()
            is_checkpoint = path.suffix in (".pt", ".safetensors", ".bin")
            kind = ("recovery_checkpoint" if path.name.startswith("recovery_") or path.name == "latest.pt"
                    else "inference_checkpoint" if path.name.startswith(("inference_", "projected_"))
                    else "binary_artifact" if is_checkpoint else "result")
            entries.append({"path": relative, "size_bytes": info.st_size,
                            "sha256": digest, "kind": kind})
            if not is_checkpoint:
                included.append(path)
    if not entries:
        raise Blocked("There are no Task 1 outputs to export")
    verification = read_json(ctx.results / "summary" / "final_verification.json", {})
    manifest = {"source_revision": ctx.revision, "files": entries,
                "completion_states": verification.get("status", {}),
                "checkpoint_files_in_zip": False,
                "durable_backup_confirmed": False,
                "instructions": "Use Kaggle Save Version; mount the resulting output dataset in the next session. "
                                "The ZIP excludes large recovery and inference checkpoints."}
    manifest_path = ctx.results / "summary" / "backup_manifest.json"
    atomic_json(manifest_path, manifest)
    included.append(manifest_path)
    included.extend([ctx.repo / "configs/task1.yaml", ctx.repo / "requirements.txt",
                     ctx.repo / "kaggle/source_manifest.json", ctx.repo / "first_plot_questions.yaml"])
    included = [path for path in included if path.is_file()]
    signature = hashlib.sha256("".join(f"{path}:{sha256_file(path)}\n" for path in included).encode()).hexdigest()
    bundle = ctx.work / "task1_results_bundle.zip"
    index = ctx.work / "task1_results_bundle.index.json"
    previous = read_json(index, {})
    if bundle.is_file() and previous.get("signature") == signature and previous.get("sha256") == sha256_file(bundle):
        print("[SKIPPED] Export  -  verified current compact bundle", flush=True)
        return {"artifacts": [bundle, manifest_path, index]}
    uncompressed = sum(path.stat().st_size for path in included)
    if shutil.disk_usage(ctx.work).free < max(uncompressed, 1024**3):
        raise Blocked("Insufficient working disk for compact result bundle; preserve outputs with Kaggle Save Version")
    temporary = bundle.with_name(bundle.name + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            for path in included:
                name = (path.relative_to(ctx.work).as_posix() if path.is_relative_to(ctx.work)
                        else "source/" + path.relative_to(ctx.repo).as_posix())
                archive.write(path, name)
        with zipfile.ZipFile(temporary) as archive:
            bad = archive.testzip()
            if bad:
                raise IOError(f"Export ZIP CRC failed at {bad}")
        os.replace(temporary, bundle)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json(index, {"signature": signature, "sha256": sha256_file(bundle),
                        "size_bytes": bundle.stat().st_size})
    print(f"[PASS] Export  -  {bundle} ({bundle.stat().st_size/2**20:.1f} MiB); "
          "full checkpoints still require Kaggle Save Version", flush=True)
    return {"artifacts": [bundle, manifest_path, index]}


def _stage_artifacts(ctx, result):
    values = result.get("artifacts", []) if isinstance(result, dict) else []
    hashes = {}
    for path in values:
        path = Path(path)
        if not path.is_file():
            raise RuntimeError(f"Stage reported a missing artifact: {path}")
        hashes[path.relative_to(ctx.work).as_posix()] = sha256_file(path)
    return hashes


def run_stage(stage, repo, work):
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage}")
    ctx = Context(repo, work)
    try:
        if stage == "environment":
            _session_seconds_remaining(ctx)
            result = stage_environment(ctx)
        elif stage == "dependencies":
            result = ensure_dependencies(ctx.repo)
            print("[PASS] Dependencies  -  clean subprocess imports and pinned versions", flush=True)
        elif stage == "restore":
            ensure_dependencies(ctx.repo)
            result = stage_restore(ctx)
        else:
            ensure_dependencies(ctx.repo)
            config = ctx.config()
            pending = ctx.state().get("backup_pending")
            if pending and stage not in ("final_verify", "export"):
                raise Blocked(f"Save Version and mount the {pending['method']} checkpoint "
                              f"at update {pending['step']} before continuing")
            if stage == "dataset":
                result = stage_dataset(ctx, config)
            elif stage == "preflight":
                result = stage_preflight(ctx, config)
            elif stage == "final_verify":
                result = stage_final_verify(ctx, config)
            elif stage == "export":
                result = stage_export(ctx, config)
            else:
                from task1.dataset import load_selected
                from task1.preflight import require_passed_preflight
                _, manifest = load_selected(config)
                require_passed_preflight(config, manifest)
                if stage == "baseline":
                    result = stage_baseline(ctx, config)
                elif stage.startswith("train_"):
                    result = stage_train(ctx, config, stage.removeprefix("train_"))
                elif stage == "evaluate_models":
                    result = stage_evaluate_models(ctx, config)
                elif stage == "weight":
                    result = stage_weight(ctx, config)
                elif stage == "activations":
                    result = stage_activations(ctx, config)
                elif stage == "qualitative":
                    result = stage_qualitative(ctx, config)
                elif stage == "aggregate":
                    result = stage_aggregate(ctx, config)
                else:
                    raise AssertionError(stage)
        ctx.record(stage, "PASS", source_revision=ctx.revision,
                   artifacts=_stage_artifacts(ctx, result),
                   detail={key: value for key, value in (result or {}).items() if key != "artifacts"}
                   if isinstance(result, dict) else result)
        return result
    except Blocked as exc:
        ctx.record(stage, "BLOCKED", source_revision=ctx.revision, reason=str(exc))
        print(f"[BLOCKED] {stage}  -  {exc}", flush=True)
        return {"blocked": str(exc)}
    except Exception as exc:
        ctx.record(stage, "FAILED", source_revision=ctx.revision,
                   error=f"{type(exc).__name__}: {exc}")
        print(f"[FAILED] {stage}  -  {type(exc).__name__}: {exc}", flush=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    run_stage(args.stage, args.repo, args.work)


if __name__ == "__main__":
    main()
