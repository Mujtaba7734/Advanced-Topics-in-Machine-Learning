from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    print("\nTASK 3 RUNTIME PRECHECK", flush=True)

    torchao_version = installed_version("torchao")
    if torchao_version is not None:
        print(
            f"  Removing unused torchao {torchao_version} "
            "to avoid PEFT dispatch conflicts...",
            flush=True,
        )
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
        installed_version(name) != expected
        for name, expected in exact.items()
    )

    bnb_version = installed_version("bitsandbytes")
    if bnb_version is None or Version(bnb_version) < Version("0.46.1"):
        needs_requirements = True

    if needs_requirements:
        print("  Repairing pinned course Python dependencies...", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(repo_path("requirements.txt")),
            ],
            check=True,
        )

    if installed_version("torchao") is not None:
        raise RuntimeError(
            "torchao is still installed after Task 3 runtime cleanup"
        )

    mismatches = {
        name: (installed_version(name), expected)
        for name, expected in exact.items()
        if installed_version(name) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"Pinned package mismatch after repair: {mismatches}"
        )

    bnb_version = installed_version("bitsandbytes")
    if bnb_version is None or Version(bnb_version) < Version("0.46.1"):
        raise RuntimeError(
            f"bitsandbytes>=0.46.1 required; found {bnb_version!r}"
        )

    print("  torchao:      absent (intentional; not used by PA2)")
    print(f"  bitsandbytes: {bnb_version}")
    for name, expected in exact.items():
        print(f"  {name}: {expected}")
    print("TASK 3 RUNTIME PRECHECK: PASS", flush=True)


def adapter_is_complete(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "adapter_config.json").exists() and any(
        (path / name).exists()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def jsonl_count(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def completed_training(
    cfg,
    name: str,
    adapter: Path,
    expected_updates: int,
    expected_loss_type: str,
    require_details: bool = False,
):
    results_dir = repo_path(cfg["results_dir"])
    summary_path = results_dir / f"{name}_train.json"
    log_path = results_dir / f"{name}_train.jsonl"

    if not (
        adapter_is_complete(adapter)
        and summary_path.exists()
        and jsonl_count(log_path) == expected_updates
    ):
        return None

    if require_details:
        expected_details = (
            expected_updates
            * int(cfg["prompts_per_update"])
            * int(cfg["num_generations"])
            * int(cfg["policy_epochs"])
        )
        details_path = results_dir / f"{name}_optimization.jsonl"
        if jsonl_count(details_path) != expected_details:
            return None

    summary = load_json(summary_path)
    checks = [
        int(summary.get("updates", -1)) == expected_updates,
        int(summary.get("num_generations", -1))
        == int(cfg["num_generations"]),
        summary.get("loss_type") == expected_loss_type,
        summary.get("midpoint")
        == cfg["paths"]["grpo_midpoint_policy"],
        abs(
            float(summary.get("clip_epsilon", -999.0))
            - float(cfg["clip_epsilon"])
        )
        < 1e-12,
        abs(
            float(summary.get("kl_beta", -999.0))
            - float(cfg["kl_beta"])
        )
        < 1e-12,
        int(summary.get("max_completion_length", -1))
        == int(cfg["max_completion_length"]),
        bool(summary.get("mask_truncated_completions", False))
        == bool(cfg["mask_truncated_completions"]),
    ]
    return summary if all(checks) else None


def completed_evaluation(cfg, name: str, expected_n: int):
    results_dir = repo_path(cfg["results_dir"])
    metrics_path = results_dir / f"{name}_eval.json"
    examples_path = results_dir / f"{name}_examples.jsonl"

    if not metrics_path.exists() or jsonl_count(examples_path) != expected_n:
        return None

    metrics = load_json(metrics_path)
    if int(metrics.get("n", -1)) != expected_n:
        return None
    if int(metrics.get("eval_batch_size", -1)) != int(
        cfg.get("eval_batch_size", 1)
    ):
        return None
    return metrics


def _all_finite(rows: list[dict], keys: list[str]) -> bool:
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value is None or not math.isfinite(float(value)):
                return False
    return True


def validate_task3_artifacts(cfg) -> None:
    results_dir = repo_path(cfg["results_dir"])
    output_dir = repo_path(cfg["output"]).parent
    eval_n = len(read_jsonl(cfg["paths"]["rl_prompt_eval"]))
    standard_updates = int(cfg["updates"])
    fork_updates = int(cfg["fork_updates"])
    k = int(cfg["num_generations"])
    checks: list[tuple[bool, str]] = []

    def require(condition, message):
        checks.append((bool(condition), message))

    standard_adapter = repo_path(cfg["output"])
    standard = completed_training(
        cfg,
        "standard",
        standard_adapter,
        standard_updates,
        "grpo",
    )
    require(
        standard is not None,
        f"standard GRPO completed exactly {standard_updates} updates "
        "from the supplied midpoint",
    )

    if standard is not None:
        require(
            float(standard.get("wall_seconds", 0.0)) > 0.0,
            "standard GRPO recorded wall-clock time",
        )
        require(
            standard.get("peak_vram_bytes") is not None,
            "standard GRPO recorded peak VRAM",
        )
        require(
            int(standard.get("max_generated_token_budget", -1))
            == standard_updates
            * int(cfg["prompts_per_update"])
            * k
            * int(cfg["max_completion_length"]),
            "standard GRPO recorded the full configured generation budget",
        )

    standard_log_path = results_dir / "standard_train.jsonl"
    standard_log = (
        read_jsonl(standard_log_path)
        if standard_log_path.exists()
        else []
    )
    required_log_keys = [
        "reward",
        "sampled_kl",
        "reward_std",
        "uninformative_group_fraction",
        "policy_term",
        "loss",
        "grad_norm",
        "sample_entropy",
        "response_length",
    ]
    require(
        len(standard_log) == standard_updates,
        f"standard GRPO log contains exactly {standard_updates} updates",
    )
    require(
        bool(standard_log)
        and all(
            all(key in row for key in required_log_keys)
            for row in standard_log
        ),
        "standard GRPO log contains every required diagnostic",
    )
    if standard_log:
        require(
            _all_finite(standard_log, required_log_keys),
            "standard GRPO required diagnostics are finite",
        )

    require(
        completed_evaluation(cfg, "standard", eval_n) is not None,
        f"standard GRPO evaluated all {eval_n} fixed held-out prompts",
    )

    group_path = results_dir / "group_size_study.json"
    require(group_path.exists(), "group_size_study.json exists")
    group_study = load_json(group_path) if group_path.exists() else {}
    generation_budgets = []

    for group_k in cfg["group_sizes"]:
        row = group_study.get(str(int(group_k)))
        require(
            row is not None,
            f"group-size diagnostic exists for K={int(group_k)}",
        )
        if row is None:
            continue

        generation_budgets.append(int(row.get("generations", -1)))
        for key in (
            "informative_group_fraction",
            "mean_reward_std",
            "mean_relative_signal_variance",
        ):
            require(
                key in row and math.isfinite(float(row[key])),
                f"K={int(group_k)} records {key}",
            )

        bins = row.get("by_prompt_difficulty", {})
        require(
            len(bins) >= 2,
            f"K={int(group_k)} reports at least two "
            "non-empty prompt-difficulty bins",
        )
        for label, values in bins.items():
            for key in (
                "informative_group_fraction",
                "mean_reward_std",
                "mean_relative_signal_variance",
            ):
                require(
                    key in values
                    and math.isfinite(float(values[key])),
                    f"K={int(group_k)} difficulty bin {label} "
                    f"records {key}",
                )

    if generation_budgets:
        require(
            len(set(generation_budgets)) == 1,
            "K=2/4/8 conditions use the same fixed generation budget",
        )

    norm_path = results_dir / "normalization_study.json"
    require(norm_path.exists(), "normalization_study.json exists")
    norm = load_json(norm_path) if norm_path.exists() else {}
    fork_budgets = []

    for loss_type in ("grpo", "dr_grpo"):
        name = f"normalization_{loss_type}"
        adapter = output_dir / name
        train = completed_training(
            cfg,
            name,
            adapter,
            fork_updates,
            loss_type,
            require_details=True,
        )
        require(
            train is not None,
            f"{name} completed exactly {fork_updates} updates "
            "from the supplied midpoint",
        )
        if train is not None:
            fork_budgets.append(
                int(train.get("max_generated_token_budget", -1))
            )

        require(
            completed_evaluation(cfg, name, eval_n) is not None,
            f"{name} evaluated all {eval_n} fixed held-out prompts",
        )

        study_row = norm.get(loss_type)
        require(
            isinstance(study_row, dict),
            f"{name} appears in normalization_study.json",
        )
        if not isinstance(study_row, dict):
            continue

        for section in (
            "length_conditioned",
            "training_length_conditioned",
        ):
            conditioned = study_row.get(section, {})
            require(
                "short" in conditioned and "long" in conditioned,
                f"{name} has shared short/long {section} statistics",
            )

        training_conditioned = study_row.get(
            "training_length_conditioned",
            {},
        )
        for label in ("short", "long"):
            values = training_conditioned.get(label, {})
            require(
                "mean_normalization_weight_per_token" in values,
                f"{name} {label} bin records normalization weight",
            )
            require(
                "mean_absolute_policy_loss_contribution" in values,
                f"{name} {label} bin records policy-loss contribution",
            )

    if fork_budgets:
        require(
            len(fork_budgets) == 2
            and len(set(fork_budgets)) == 1,
            "canonical GRPO and Dr-GRPO use equal configured "
            "generated-token budgets",
        )

    failures = [message for ok, message in checks if not ok]
    if failures:
        formatted = "\n  - ".join(failures)
        raise RuntimeError(
            "Task 3 completeness validation FAILED. Nothing will be "
            "declared complete or backed up.\n  - "
            + formatted
        )

    group_budget = generation_budgets[0] if generation_budgets else 0

    print("\nTASK 3 COMPLETENESS CHECK: PASS")
    print(
        f"  Standard continuation:      "
        f"{standard_updates}/{standard_updates} updates, K={k}"
    )
    print(
        f"  Standard held-out eval:     {eval_n}/{eval_n} prompts"
    )
    print(
        f"  Group-size cached study:    "
        f"K={list(cfg['group_sizes'])}, "
        f"{group_budget} fixed generations each"
    )
    print(
        f"  Normalization forks:        "
        f"2 x {fork_updates} updates from same midpoint"
    )
    print(
        f"  Normalization evals:        2 x {eval_n} prompts"
    )
    print("  Length-conditioned optimization statistics recorded.")
    print("  Standard wall-clock and peak VRAM recorded.")
    print("  No Task 3 experimental budget was shortened.", flush=True)


def create_verified_backup(cfg) -> Path:
    repo_root = repo_path(".")
    results_dir = repo_path(cfg["results_dir"])
    output_dir = repo_path(cfg["output"]).parent
    config_path = repo_path("configs/grpo.yaml")
    backup_path = Path(
        "/kaggle/working/ATML_PA2_Task3_BACKUP.tar.gz"
    )
    checksum_path = Path(str(backup_path) + ".sha256")
    manifest_path = results_dir / "task3_backup_manifest.json"

    required = [results_dir, output_dir, config_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Refusing to create Task 3 backup because inputs are missing: "
            + ", ".join(missing)
        )

    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
    except Exception:
        git_head = None

    manifest = {
        "task": "task3_grpo",
        "git_head": git_head,
        "results_dir": str(results_dir),
        "outputs_dir": str(output_dir),
        "config": str(config_path),
        "backup_path": str(backup_path),
        "standard_policy_adapter": str(repo_path(cfg["output"])),
        "group_sizes": [int(k) for k in cfg["group_sizes"]],
        "normalization_conditions": ["grpo", "dr_grpo"],
        "eval_batch_size": int(cfg.get("eval_batch_size", 1)),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    with tarfile.open(backup_path, "w:gz") as tar:
        for path in (results_dir, output_dir, config_path):
            resolved = path.resolve()
            arcname = resolved.relative_to(repo_root.resolve())
            tar.add(resolved, arcname=str(arcname))

    with tarfile.open(backup_path, "r:gz") as tar:
        names = tar.getnames()
        for required_prefix in (
            str(Path(cfg["results_dir"])),
            str(Path(cfg["output"]).parent),
        ):
            if not any(
                name == required_prefix
                or name.startswith(required_prefix + "/")
                for name in names
            ):
                raise RuntimeError(
                    "Task 3 backup verification failed: missing "
                    f"{required_prefix}"
                )

    sha256 = hashlib.sha256()
    with backup_path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    digest = sha256.hexdigest()
    checksum_path.write_text(
        f"{digest}  {backup_path.name}\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("TASK 3 BACKUP CREATED AND VERIFIED")
    print(f"Backup:   {backup_path}")
    print(f"SHA256:   {digest}")
    print(f"Checksum: {checksum_path}")
    print(
        f"Size MiB: {backup_path.stat().st_size / (1024 ** 2):.1f}"
    )
    print(
        "Download the archive and checksum before stopping/resetting "
        "the Kaggle session."
    )
    print("=" * 80, flush=True)
    return backup_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the complete Task 3 GRPO experiment pipeline."
    )
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse complete Task 3 stages and resume interrupted "
            "held-out evaluation batches."
        ),
    )
    args = ap.parse_args()

    ensure_runtime_compatibility()
    run_module("scripts.check_environment")
    run_module("scripts.validate_assets")
    run_module("task3_grpo.self_test")

    # This required study is cache-only and cheap. Run it before any training
    # so cache/schema/regrouping problems fail before GPU time is spent.
    run_module(
        "task3_grpo.analyze_group_size",
        "--config",
        args.config,
    )

    cfg = load_yaml(args.config)
    standard_adapter = repo_path(cfg["output"])
    eval_n = len(read_jsonl(cfg["paths"]["rl_prompt_eval"]))

    standard_train = (
        completed_training(
            cfg,
            "standard",
            standard_adapter,
            int(cfg["updates"]),
            "grpo",
        )
        if args.resume
        else None
    )
    if standard_train is not None:
        print(
            "[resume] Reusing completed standard GRPO continuation",
            flush=True,
        )
    else:
        run_module(
            "task3_grpo.continue_train",
            "--config",
            args.config,
            "--run-name",
            "standard",
            "--output",
            str(standard_adapter),
            "--loss-type",
            "grpo",
        )

    standard_eval = (
        completed_evaluation(cfg, "standard", eval_n)
        if args.resume
        else None
    )
    if standard_eval is not None:
        print(
            "[resume] Reusing completed standard GRPO evaluation",
            flush=True,
        )
    else:
        eval_args = [
            "--config",
            args.config,
            "--adapter",
            str(standard_adapter),
            "--name",
            "standard",
        ]
        if args.resume:
            eval_args.append("--resume")
        run_module("task3_grpo.evaluate", *eval_args)

    normalization_args = ["--config", args.config]
    if args.resume:
        normalization_args.append("--resume")
    run_module(
        "task3_grpo.compare_normalization",
        *normalization_args,
    )

    validate_task3_artifacts(cfg)
    backup_path = create_verified_backup(cfg)

    print("\nTask 3 pipeline complete and verified.", flush=True)
    print(f"Results: {repo_path(cfg['results_dir'])}", flush=True)
    print(
        f"Standard GRPO policy adapter: {standard_adapter}",
        flush=True,
    )
    print(f"Verified backup: {backup_path}", flush=True)


if __name__ == "__main__":
    main()
