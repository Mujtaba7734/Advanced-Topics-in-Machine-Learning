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
from task5_feedback.rlaif import JUDGE_PROTOCOL_VERSION
from task5_feedback.evaluate_math import policy_specs


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
    print("\nTASK 5 RUNTIME PRECHECK", flush=True)

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
        raise RuntimeError("torchao is still installed after Task 5 runtime cleanup")

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
    print("TASK 5 RUNTIME PRECHECK: PASS", flush=True)


def adapter_is_complete(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "adapter_config.json").exists() and any(
        (path / name).exists()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def preflight_policy_checkpoints(cfg) -> None:
    specs = policy_specs(cfg)
    expected = {
        "sft": None,
        "rlvr": "checkpoints/rlvr_policy",
        "rlaif": "checkpoints/rlaif_policy",
    }

    if specs != expected:
        raise ValueError(
            f"Task 5 must use the fixed SFT/RLVR/RLAIF policies. "
            f"Expected {expected}, found {specs}"
        )

    failures = []
    for name in ("rlvr", "rlaif"):
        path = repo_path(specs[name])
        if not adapter_is_complete(path):
            failures.append(f"{name}: {path}")

    if failures:
        raise FileNotFoundError(
            "Task 5 supplied policy adapters are missing/incomplete:\n  - "
            + "\n  - ".join(failures)
        )

    print("TASK 5 POLICY CHECKPOINT PREFLIGHT: PASS")
    print("  SFT:   untouched base Qwen/Qwen2.5-1.5B-Instruct")
    print(f"  RLVR:  {specs['rlvr']}")
    print(f"  RLAIF: {specs['rlaif']}", flush=True)


def _source_id(row: dict, position: int) -> str:
    return str(row.get("source_index", row.get("prompt_id", position)))


def _jsonl(path: Path) -> list[dict]:
    return read_jsonl(path) if path.exists() else []


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def validate_task5_artifacts(cfg) -> None:
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    gsm_rows = read_jsonl(cfg["paths"]["gsm_eval"])
    transfer_rows = read_jsonl(cfg["paths"]["math_transfer_eval"])
    expected_by_dataset = {
        "gsm": gsm_rows,
        "transfer": transfer_rows,
    }
    policies = ("sft", "rlvr", "rlaif")

    checks: list[tuple[bool, str]] = []

    def require(condition, message):
        checks.append((bool(condition), message))

    require(len(transfer_rows) == 100, "transfer evaluation uses exactly 100 fixed SVAMP examples")

    for dataset, rows in expected_by_dataset.items():
        expected_ids = [_source_id(row, i) for i, row in enumerate(rows)]
        n = len(rows)
        require(n > 0, f"{dataset} fixed evaluation set is non-empty")

        metrics_path = outdir / f"{dataset}_metrics.json"
        pairwise_path = outdir / f"{dataset}_pairwise.jsonl"
        require(metrics_path.exists(), f"{dataset}_metrics.json exists")
        require(pairwise_path.exists(), f"{dataset}_pairwise.jsonl exists")

        metrics = load_json(metrics_path) if metrics_path.exists() else {}
        require(int(metrics.get("n", -1)) == n, f"{dataset} metrics cover all {n} examples")
        require(
            int(metrics.get("judge_protocol_version", -1)) == JUDGE_PROTOCOL_VERSION,
            f"{dataset} metrics use judge protocol v{JUDGE_PROTOCOL_VERSION}",
        )

        for policy in policies:
            response_path = outdir / f"{dataset}_{policy}_responses.jsonl"
            responses = _jsonl(response_path)
            require(len(responses) == n, f"{dataset}/{policy} has all {n} responses")
            if len(responses) == n:
                ids = [str(row.get("source_index")) for row in responses]
                require(ids == expected_ids, f"{dataset}/{policy} preserves fixed source ID/order")
                require(
                    all(row.get("policy") == policy for row in responses),
                    f"{dataset}/{policy} response metadata is consistent",
                )
                require(
                    all("response" in row and "response_tokens" in row for row in responses),
                    f"{dataset}/{policy} stores response text and token length",
                )

            policy_metrics = metrics.get("policies", {}).get(policy, {})
            for key in (
                "exact_accuracy",
                "format_compliance",
                "mean_response_tokens",
                "response_tokens_std",
            ):
                require(
                    key in policy_metrics and _finite(policy_metrics.get(key)),
                    f"{dataset}/{policy} records finite {key}",
                )

        comparisons = _jsonl(pairwise_path)
        expected_pairs = n * 3
        require(
            len(comparisons) == expected_pairs,
            f"{dataset} has exactly {expected_pairs} pairwise judgments",
        )
        if comparisons:
            require(
                all(row.get("preference") in {"A", "B", "TIE"} for row in comparisons),
                f"{dataset} pairwise labels are valid",
            )
            require(
                all(
                    int(row.get("judge_protocol_version", 0)) == JUDGE_PROTOCOL_VERSION
                    for row in comparisons
                ),
                f"{dataset} pairwise judgments use corrected judge protocol",
            )

        for pair_name in ("sft_vs_rlvr", "sft_vs_rlaif", "rlvr_vs_rlaif"):
            pair_metrics = metrics.get("pairwise", {}).get(pair_name, {})
            require(
                int(pair_metrics.get("n", -1)) == n,
                f"{dataset}/{pair_name} covers all {n} examples",
            )
            for key in (
                "a_pairwise_score",
                "b_pairwise_score",
                "tie_rate",
                "verifier_judge_strict_three_way_agreement",
            ):
                require(
                    key in pair_metrics and _finite(pair_metrics.get(key)),
                    f"{dataset}/{pair_name} records finite {key}",
                )

    perturb_path = outdir / "perturbation_metrics.json"
    pairs_path = outdir / "perturbation_pairs.jsonl"
    require(perturb_path.exists(), "perturbation_metrics.json exists")
    require(pairs_path.exists(), "perturbation_pairs.jsonl exists")

    perturb = load_json(perturb_path) if perturb_path.exists() else {}
    require(int(perturb.get("n_problems", -1)) == 20, "diagnostic study uses all 20 fixed problems")
    require(int(perturb.get("n_responses", -1)) == 100, "diagnostic study uses all 100 fixed responses")
    require(int(perturb.get("n_pairs", -1)) == 80, "diagnostic study scores all 80 clean-vs-perturbation pairs")
    require(
        int(perturb.get("judge_protocol_version", -1)) == JUDGE_PROTOCOL_VERSION,
        "diagnostic study uses corrected judge protocol",
    )

    categories = perturb.get("categories", {})
    expected_categories = (
        "clean_correct",
        "corrupt_reasoning_correct_final",
        "good_reasoning_wrong_final",
        "persuasive_filler_correct",
        "gold_distractor_wrong_final",
    )
    for category in expected_categories:
        require(category in categories, f"diagnostic metrics contain category {category}")

    for category in expected_categories[1:]:
        row = categories.get(category, {})
        require(int(row.get("n", -1)) == 20, f"{category} contains 20 matched pairs")
        for mechanism in ("rlvr", "rlaif"):
            rates = row.get(mechanism, {})
            for key in (
                "better_response_rate",
                "tie_rate",
                "wrong_preference_rate",
                "pairwise_score",
            ):
                require(
                    key in rates and _finite(rates.get(key)),
                    f"{category}/{mechanism} records finite {key}",
                )

    for key in ("S_reason", "S_outcome"):
        row = perturb.get(key, {})
        for mechanism in ("rlvr", "rlaif"):
            require(
                mechanism in row and _finite(row.get(mechanism)),
                f"{key} records finite {mechanism} sensitivity",
            )

    comparison_path = outdir / "feedback_comparison.json"
    require(comparison_path.exists(), "feedback_comparison.json exists")
    if comparison_path.exists():
        comparison = load_json(comparison_path)
        for key in (
            "exact_accuracy_drop_gsm_minus_transfer",
            "response_length_shift_transfer_minus_gsm",
            "pairwise_score_drop_gsm_minus_transfer",
        ):
            require(key in comparison, f"final Task 5 comparison records {key}")

    failures = [message for ok, message in checks if not ok]
    if failures:
        raise RuntimeError(
            "Task 5 completeness validation FAILED.\n  - "
            + "\n  - ".join(failures)
        )

    print("\nTASK 5 COMPUTATIONAL COMPLETENESS CHECK: PASS")
    print(f"  GSM8K examples:             {len(gsm_rows)}")
    print("  Frozen policies:            SFT / RLVR / RLAIF")
    print("  In-domain pairwise judging: PASS")
    print("  Diagnostic responses:       100")
    print("  Diagnostic matched pairs:   80")
    print("  SVAMP transfer examples:    100")
    print("  Transfer-drop metrics:      PASS")
    print("  No Task 5 training was performed.", flush=True)


def create_verified_backup(cfg) -> Path:
    repo_root = repo_path(".")
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    config_path = repo_path("configs/feedback.yaml")
    backup_path = Path("/kaggle/working/ATML_PA2_Task5_BACKUP.tar.gz")
    checksum_path = Path(str(backup_path) + ".sha256")
    manifest_path = outdir / "task5_backup_manifest.json"

    if not outdir.exists():
        raise FileNotFoundError(f"Task 5 results directory is missing: {outdir}")

    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
    except Exception:
        git_head = None

    manifest = {
        "task": "task5_feedback",
        "git_head": git_head,
        "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
        "results_dir": str(outdir),
        "config": str(config_path),
        "backup_path": str(backup_path),
        "training_performed": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    archive_inputs = [outdir, config_path, manifest_path]
    seen = set()
    with tarfile.open(backup_path, "w:gz") as tar:
        for path in archive_inputs:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                arcname = resolved.relative_to(repo_root.resolve())
            except ValueError:
                arcname = Path(resolved.name)
            tar.add(resolved, arcname=str(arcname))

    with tarfile.open(backup_path, "r:gz") as tar:
        names = tar.getnames()
        results_prefix = str(
            outdir.resolve().relative_to(repo_root.resolve())
        )
        if not any(
            name == results_prefix or name.startswith(results_prefix + "/")
            for name in names
        ):
            raise RuntimeError("Task 5 backup verification failed: results missing")
        if str(config_path.resolve().relative_to(repo_root.resolve())) not in names:
            raise RuntimeError("Task 5 backup verification failed: config missing")

    sha256 = hashlib.sha256()
    with backup_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
    digest = sha256.hexdigest()
    checksum_path.write_text(
        f"{digest}  {backup_path.name}\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("TASK 5 BACKUP CREATED AND VERIFIED")
    print(f"Backup:   {backup_path}")
    print(f"SHA256:   {digest}")
    print(f"Checksum: {checksum_path}")
    print(f"Size MiB: {backup_path.stat().st_size / (1024 ** 2):.1f}")
    print("Download the archive and checksum before deleting the Kaggle session.")
    print("=" * 80, flush=True)
    return backup_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the complete frozen-policy Task 5 evaluation pipeline."
    )
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()

    ensure_runtime_compatibility()
    run_module("scripts.check_environment")
    run_module("scripts.validate_assets")
    run_module("task5_feedback.self_test")

    cfg = load_yaml(args.config)
    preflight_policy_checkpoints(cfg)

    run_module(
        "task5_feedback.evaluate_math",
        "--config", args.config,
        "--dataset", "gsm",
    )
    run_module(
        "task5_feedback.score_perturbations",
        "--config", args.config,
    )
    run_module(
        "task5_feedback.evaluate_math",
        "--config", args.config,
        "--dataset", "transfer",
    )
    run_module(
        "task5_feedback.compare_feedback",
        "--config", args.config,
    )

    validate_task5_artifacts(cfg)
    backup_path = create_verified_backup(cfg)

    print("\nTask 5 GPU-dependent evaluation is complete and verified.")
    print(f"Verified backup: {backup_path}")
    print("After downloading the backup + checksum, PA2 no longer needs Kaggle GPU compute.", flush=True)


if __name__ == "__main__":
    main()
