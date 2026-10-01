from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from importlib import metadata
from pathlib import Path

from packaging.version import Version

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json


def run_module(module: str, *args: str) -> None:
    command = [sys.executable, "-m", module, *args]
    print("\n" + "=" * 80)
    print("Running:", " ".join(command))
    print("=" * 80, flush=True)
    subprocess.run(command, check=True)


def installed_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def ensure_runtime_compatibility() -> None:
    """Repair known Kaggle package conflicts before importing PEFT in child jobs."""
    print("\nTASK 2 RUNTIME PRECHECK", flush=True)

    # Kaggle images may ship torchao 0.10.0. PEFT 0.17.1 detects the optional
    # package during LoRA injection and rejects that old version even though
    # this assignment does not use torchao at all. Remove it deliberately.
    torchao_version = installed_version("torchao")
    if torchao_version is not None:
        print(f"  Removing unused torchao {torchao_version} to avoid PEFT dispatch conflicts...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
            check=True,
        )

    exact = {
        "transformers": "4.57.1",
        "tokenizers": "0.22.1",
        "peft": "0.17.1",
        "trl": "0.27.2",
    }
    needs_requirements = any(
        installed_version(name) != expected for name, expected in exact.items()
    )

    bnb_version = installed_version("bitsandbytes")
    if bnb_version is None or Version(bnb_version) < Version("0.46.1"):
        needs_requirements = True

    if needs_requirements:
        print("  Repairing pinned course Python dependencies...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(repo_path("requirements.txt"))],
            check=True,
        )

    if installed_version("torchao") is not None:
        raise RuntimeError("torchao is still installed after Task 2 runtime cleanup")

    mismatches = {
        name: (installed_version(name), expected)
        for name, expected in exact.items()
        if installed_version(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"Pinned package mismatch after repair: {mismatches}")

    bnb_version = installed_version("bitsandbytes")
    if bnb_version is None or Version(bnb_version) < Version("0.46.1"):
        raise RuntimeError(
            f"bitsandbytes>=0.46.1 required; found {bnb_version!r}"
        )

    print("  torchao:      absent (intentional; not used by PA2)")
    print(f"  bitsandbytes: {bnb_version}")
    for name, expected in exact.items():
        print(f"  {name}: {expected}")
    print("TASK 2 RUNTIME PRECHECK: PASS", flush=True)


def adapter_is_complete(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "adapter_config.json").exists()
    has_weights = any((path / name).exists() for name in (
        "adapter_model.safetensors", "adapter_model.bin"
    ))
    return has_config and has_weights


def jsonl_count(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def completed_training(cfg, name: str, adapter: Path, expected_updates: int):
    value_adapter = adapter.parent / f"{adapter.name}_value"
    summary_path = repo_path(cfg["results_dir"]) / f"{name}_train.json"
    log_path = repo_path(cfg["results_dir"]) / f"{name}_train.jsonl"
    if not (
        adapter_is_complete(adapter)
        and adapter_is_complete(value_adapter)
        and summary_path.exists()
        and jsonl_count(log_path) == expected_updates
    ):
        return None
    summary = load_json(summary_path)
    if int(summary.get("updates", -1)) != expected_updates:
        return None
    return summary


def completed_evaluation(cfg, name: str, expected_n: int):
    metrics_path = repo_path(cfg["results_dir"]) / f"{name}_eval.json"
    examples_path = repo_path(cfg["results_dir"]) / f"{name}_examples.jsonl"
    if not metrics_path.exists() or jsonl_count(examples_path) != expected_n:
        return None
    metrics = load_json(metrics_path)
    if int(metrics.get("n", -1)) != expected_n:
        return None
    return metrics


def validate_task2_artifacts(cfg) -> None:
    results_dir = repo_path(cfg["results_dir"])
    output_dir = repo_path(cfg["output"]).parent
    eval_n = len(read_jsonl(cfg["paths"]["rl_prompt_eval"]))
    standard_updates = int(cfg["updates"])
    fork_updates = int(cfg["fork_updates"])
    checks: list[tuple[bool, str]] = []

    def require(condition, message):
        checks.append((bool(condition), message))

    # Standard PPO continuation.
    standard_adapter = repo_path(cfg["output"])
    standard_value = standard_adapter.parent / f"{standard_adapter.name}_value"
    standard_summary_path = results_dir / "standard_train.json"
    standard_log_path = results_dir / "standard_train.jsonl"
    require(adapter_is_complete(standard_adapter), "standard PPO policy adapter exists")
    require(adapter_is_complete(standard_value), "standard PPO value adapter exists")
    require(standard_summary_path.exists(), "standard_train.json exists")
    require(jsonl_count(standard_log_path) == standard_updates,
            f"standard PPO log contains exactly {standard_updates} updates")
    if standard_summary_path.exists():
        summary = load_json(standard_summary_path)
        require(int(summary.get("updates", -1)) == standard_updates,
                f"standard PPO ran exactly {standard_updates} updates")
        require(abs(float(summary.get("clip_epsilon", -999.0)) - float(cfg["clip_epsilon"])) < 1e-12,
                f"standard PPO used clip epsilon {float(cfg['clip_epsilon']):.2f}")
        require(abs(float(summary.get("kl_beta", -999.0)) - float(cfg["kl_beta"])) < 1e-12,
                f"standard PPO used KL beta {float(cfg['kl_beta']):.2f}")
        require(summary.get("midpoint_policy") == cfg["paths"]["ppo_midpoint_policy"],
                "standard PPO started from supplied midpoint policy")
        require(summary.get("midpoint_value") == cfg["paths"]["ppo_midpoint_value"],
                "standard PPO started from supplied midpoint value model")
        require(float(summary.get("wall_seconds", 0.0)) > 0.0,
                "standard PPO recorded wall-clock time")
        require(summary.get("peak_vram_bytes") is not None,
                "standard PPO recorded peak VRAM")

    standard_eval = completed_evaluation(cfg, "standard", eval_n)
    require(standard_eval is not None,
            f"standard PPO evaluated all {eval_n} fixed held-out prompts")

    # Clipping study.
    clipping_path = results_dir / "clipping_study.json"
    require(clipping_path.exists(), "clipping_study.json exists")
    clipping = load_json(clipping_path) if clipping_path.exists() else {}
    cached = clipping.get("cached", {})
    forks = clipping.get("forks", [])
    for eps in cfg["clip_values"]:
        key = str(eps)
        cached_row = cached.get(key)
        if cached_row is None:
            # JSON stringification may represent 0.20 as 0.2; compare numerically.
            cached_row = next(
                (value for k, value in cached.items() if abs(float(k) - float(eps)) < 1e-12),
                None,
            )
        require(cached_row is not None, f"cached clipping diagnostic exists for epsilon={float(eps):.2f}")
        if cached_row is not None:
            require(int(cached_row.get("n_rollouts", 0)) > 0,
                    f"cached clipping diagnostic has rollouts for epsilon={float(eps):.2f}")
            require(int(cached_row.get("tokens", 0)) > 0,
                    f"cached clipping diagnostic has tokens for epsilon={float(eps):.2f}")
            require("affected_token_fraction" in cached_row,
                    f"cached affected-token fraction recorded for epsilon={float(eps):.2f}")

        name = f"clip_{float(eps):.2f}"
        adapter = output_dir / name
        train = completed_training(cfg, name, adapter, fork_updates)
        require(train is not None, f"{name} completed exactly {fork_updates} updates")
        if train is not None:
            require(abs(float(train.get("clip_epsilon", -999.0)) - float(eps)) < 1e-12,
                    f"{name} used epsilon={float(eps):.2f}")
            require(abs(float(train.get("kl_beta", -999.0)) - float(cfg["kl_beta"])) < 1e-12,
                    f"{name} kept KL beta fixed at {float(cfg['kl_beta']):.2f}")
            require(train.get("midpoint_policy") == cfg["paths"]["ppo_midpoint_policy"],
                    f"{name} started from supplied midpoint policy")
            require(train.get("midpoint_value") == cfg["paths"]["ppo_midpoint_value"],
                    f"{name} started from supplied midpoint value model")
        require(completed_evaluation(cfg, name, eval_n) is not None,
                f"{name} evaluated all {eval_n} fixed held-out prompts")

        fork_row = next(
            (row for row in forks if abs(float(row.get("clip_epsilon", -999.0)) - float(eps)) < 1e-12),
            None,
        )
        require(fork_row is not None, f"{name} appears in clipping_study.json")
        if fork_row is not None:
            stability = fork_row.get("stability", {})
            require(int(stability.get("n_updates", -1)) == fork_updates,
                    f"{name} stability statistic uses all {fork_updates} updates")
            require(stability.get("policy_grad_norm_std") is not None,
                    f"{name} records policy-grad stability statistic")

    # KL-pressure study.
    kl_path = results_dir / "kl_study.json"
    require(kl_path.exists(), "kl_study.json exists")
    kl_rows = load_json(kl_path) if kl_path.exists() else []
    for beta in cfg["kl_values"]:
        name = f"kl_{float(beta):.2f}"
        adapter = output_dir / name
        train = completed_training(cfg, name, adapter, fork_updates)
        require(train is not None, f"{name} completed exactly {fork_updates} updates")
        if train is not None:
            require(abs(float(train.get("kl_beta", -999.0)) - float(beta)) < 1e-12,
                    f"{name} used KL beta={float(beta):.2f}")
            require(abs(float(train.get("clip_epsilon", -999.0)) - float(cfg["clip_epsilon"])) < 1e-12,
                    f"{name} kept clip epsilon fixed at {float(cfg['clip_epsilon']):.2f}")
            require(train.get("midpoint_policy") == cfg["paths"]["ppo_midpoint_policy"],
                    f"{name} started from supplied midpoint policy")
            require(train.get("midpoint_value") == cfg["paths"]["ppo_midpoint_value"],
                    f"{name} started from supplied midpoint value model")
        require(completed_evaluation(cfg, name, eval_n) is not None,
                f"{name} evaluated all {eval_n} fixed held-out prompts")
        require(any(
            abs(float(row.get("kl_beta", -999.0)) - float(beta)) < 1e-12
            for row in kl_rows
        ), f"{name} appears in kl_study.json")

    failures = [message for ok, message in checks if not ok]
    if failures:
        formatted = "\n  - ".join(failures)
        raise RuntimeError(
            "Task 2 completeness validation FAILED. Nothing will be declared complete or backed up.\n"
            f"  - {formatted}"
        )

    print("\nTASK 2 COMPLETENESS CHECK: PASS")
    print(f"  Standard continuation: {standard_updates}/{standard_updates} updates")
    print(f"  Standard held-out eval: {eval_n}/{eval_n} prompts")
    print(f"  Clipping cached study:  epsilon = {list(cfg['clip_values'])}")
    print(f"  Clipping forks:         {len(cfg['clip_values'])} x {fork_updates} updates")
    print(f"  Clipping evals:         {len(cfg['clip_values'])} x {eval_n} prompts")
    print(f"  KL forks:               {len(cfg['kl_values'])} x {fork_updates} updates")
    print(f"  KL evals:               {len(cfg['kl_values'])} x {eval_n} prompts")
    print("  Standard wall-clock and peak VRAM recorded.")
    print("  No Task 2 experimental budget was shortened.", flush=True)


def create_verified_backup(cfg) -> Path:
    repo_root = repo_path(".")
    results_dir = repo_path(cfg["results_dir"])
    output_dir = repo_path(cfg["output"]).parent
    config_path = repo_path("configs/ppo.yaml")
    backup_path = Path("/kaggle/working/ATML_PA2_Task2_BACKUP.tar.gz")
    checksum_path = Path(str(backup_path) + ".sha256")
    manifest_path = results_dir / "task2_backup_manifest.json"

    required = [results_dir, output_dir, config_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Refusing to create Task 2 backup because inputs are missing: "
            + ", ".join(missing)
        )

    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        git_head = None

    manifest = {
        "task": "task2_ppo",
        "git_head": git_head,
        "results_dir": str(results_dir),
        "outputs_dir": str(output_dir),
        "config": str(config_path),
        "backup_path": str(backup_path),
        "standard_policy_adapter": str(repo_path(cfg["output"])),
        "standard_value_adapter": str(
            repo_path(cfg["output"]).parent / f"{repo_path(cfg['output']).name}_value"
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with tarfile.open(backup_path, "w:gz") as tar:
        for path in (results_dir, output_dir, config_path):
            resolved = path.resolve()
            arcname = resolved.relative_to(repo_root.resolve())
            tar.add(resolved, arcname=str(arcname))

    with tarfile.open(backup_path, "r:gz") as tar:
        names = tar.getnames()
        for required_prefix in (str(Path(cfg["results_dir"])), str(Path(cfg["output"]).parent)):
            if not any(
                name == required_prefix or name.startswith(required_prefix + "/")
                for name in names
            ):
                raise RuntimeError(
                    f"Task 2 backup verification failed: missing {required_prefix}"
                )

    sha256 = hashlib.sha256()
    with backup_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
    digest = sha256.hexdigest()
    checksum_path.write_text(f"{digest}  {backup_path.name}\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print("TASK 2 BACKUP CREATED AND VERIFIED")
    print(f"Backup:   {backup_path}")
    print(f"SHA256:   {digest}")
    print(f"Checksum: {checksum_path}")
    print(f"Size MiB: {backup_path.stat().st_size / (1024 ** 2):.1f}")
    print("Preserve this backup with Kaggle Save Version/output or download it")
    print("before stopping/resetting the session.")
    print("=" * 80, flush=True)
    return backup_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the complete Task 2 PPO experiment pipeline.")
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse complete Task 2 stages and continue from the first incomplete stage.")
    args = ap.parse_args()

    ensure_runtime_compatibility()
    run_module("scripts.check_environment")
    run_module("scripts.validate_assets")
    run_module("task2_ppo.self_test")

    cfg = load_yaml(args.config)
    standard_adapter = repo_path(cfg["output"])
    eval_n = len(read_jsonl(cfg["paths"]["rl_prompt_eval"]))

    standard_train = completed_training(
        cfg, "standard", standard_adapter, int(cfg["updates"])
    ) if args.resume else None
    if standard_train is not None:
        print("[resume] Reusing completed standard PPO continuation", flush=True)
    else:
        run_module(
            "task2_ppo.continue_train",
            "--config", args.config,
            "--run-name", "standard",
            "--output", str(standard_adapter),
        )

    standard_eval = completed_evaluation(cfg, "standard", eval_n) if args.resume else None
    if standard_eval is not None:
        print("[resume] Reusing completed standard PPO evaluation", flush=True)
    else:
        run_module(
            "task2_ppo.evaluate",
            "--config", args.config,
            "--adapter", str(standard_adapter),
            "--name", "standard",
        )

    clip_args = ["--config", args.config]
    if args.resume:
        clip_args.append("--resume")
    run_module("task2_ppo.analyze_clipping", *clip_args)

    kl_args = ["--config", args.config]
    if args.resume:
        kl_args.append("--resume")
    run_module("task2_ppo.ablate_kl", *kl_args)

    validate_task2_artifacts(cfg)
    backup_path = create_verified_backup(cfg)

    print("\nTask 2 pipeline complete and verified.", flush=True)
    print(f"Results: {repo_path(cfg['results_dir'])}", flush=True)
    print(f"Standard policy adapter: {standard_adapter}", flush=True)
    print(f"Verified backup: {backup_path}", flush=True)


if __name__ == "__main__":
    main()
