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

import pandas as pd
from packaging.version import Version

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json
from task4_safety.judge_responses import LABELS
from task4_safety.generate_responses import POLICY_ORDER, policy_specs


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
    print("\nTASK 4 RUNTIME PRECHECK", flush=True)

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
            "torchao is still installed after Task 4 runtime cleanup"
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
    print("TASK 4 RUNTIME PRECHECK: PASS", flush=True)


def adapter_is_complete(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "adapter_config.json").exists() and any(
        (path / name).exists()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def preflight_policy_checkpoints(cfg) -> None:
    specs = policy_specs(cfg)
    failures = []

    for name in ("dpo", "ppo", "grpo"):
        adapter = repo_path(specs[name])
        if not adapter_is_complete(adapter):
            failures.append(f"{name}: {adapter}")

    if failures:
        raise FileNotFoundError(
            "Task 4 requires the STANDARD Task 1-3 policy adapters before "
            "any GPU evaluation begins. Missing/incomplete:\n  - "
            + "\n  - ".join(failures)
        )

    expected = {
        "dpo": "outputs/task1_dpo/standard",
        "ppo": "outputs/task2_ppo/standard",
        "grpo": "outputs/task3_grpo/standard",
    }
    for name, expected_path in expected.items():
        if str(specs[name]) != expected_path:
            raise ValueError(
                f"Task 4 {name} policy must be {expected_path}, "
                f"not {specs[name]!r}"
            )

    print("TASK 4 POLICY CHECKPOINT PREFLIGHT: PASS")
    print("  SFT:  untouched base Qwen/Qwen2.5-1.5B-Instruct")
    print(f"  DPO:  {specs['dpo']}")
    print(f"  PPO:  {specs['ppo']}")
    print(f"  GRPO: {specs['grpo']}", flush=True)


def _expected_xstest(cfg):
    df = pd.read_csv(repo_path(cfg["paths"]["xstest"]))
    return df


def validate_compute_artifacts(cfg) -> None:
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    xstest = _expected_xstest(cfg)
    expected_ids = xstest["xstest_id"].astype(int).tolist()
    expected_n = len(expected_ids)

    checks: list[tuple[bool, str]] = []

    def require(condition, message):
        checks.append((bool(condition), message))

    baseline_meta = None

    for policy in POLICY_ORDER:
        generated_path = outdir / f"generated_{policy}.jsonl"
        judged_path = outdir / f"judged_{policy}.jsonl"

        generated = (
            read_jsonl(generated_path)
            if generated_path.exists()
            else []
        )
        judged = (
            read_jsonl(judged_path)
            if judged_path.exists()
            else []
        )

        require(
            len(generated) == expected_n,
            f"{policy} generated all {expected_n} XSTest responses",
        )
        require(
            len(judged) == expected_n,
            f"{policy} received all {expected_n} categorical judgments",
        )

        generated_ids = [
            int(row["xstest_id"])
            for row in generated
        ]
        judged_ids = [
            int(row["xstest_id"])
            for row in judged
        ]
        require(
            generated_ids == expected_ids,
            f"{policy} generation preserved fixed XSTest ID/order",
        )
        require(
            judged_ids == expected_ids,
            f"{policy} judging preserved fixed XSTest ID/order",
        )

        if len(judged) == expected_n:
            require(
                all(row.get("label") in LABELS for row in judged),
                f"{policy} judge labels are all valid",
            )
            require(
                all(
                    math.isfinite(float(row.get("confidence", float("nan"))))
                    and 0.0 <= float(row["confidence"]) <= 1.0
                    for row in judged
                ),
                f"{policy} judge confidences are finite and in [0,1]",
            )

            metadata = [
                (
                    int(row["xstest_id"]),
                    str(row["prompt"]),
                    str(row["benchmark_class"]).upper(),
                    str(row["type"]),
                )
                for row in judged
            ]
            if baseline_meta is None:
                baseline_meta = metadata
            else:
                require(
                    metadata == baseline_meta,
                    f"{policy} used the same fixed prompts/categories as SFT",
                )

    ids_path = outdir / "manual_audit_ids.csv"
    sheet_path = outdir / "manual_audit_sheet.csv"
    require(ids_path.exists(), "fixed manual-audit ID file exists")
    require(sheet_path.exists(), "blind manual-audit sheet exists")

    if ids_path.exists():
        audit_ids = pd.read_csv(ids_path)["xstest_id"].astype(int).tolist()
        require(
            len(audit_ids) == int(cfg["manual_audit_per_class"]) * 2,
            "manual audit contains the configured 60 fixed prompts",
        )
        require(
            len(audit_ids) == len(set(audit_ids)),
            "manual audit IDs are unique",
        )
        classes = {
            int(row["xstest_id"]): str(row["benchmark_class"]).upper()
            for row in read_jsonl(outdir / "generated_sft.jsonl")
        }
        safe_n = sum(classes[sid] == "SAFE" for sid in audit_ids)
        unsafe_n = sum(classes[sid] == "UNSAFE" for sid in audit_ids)
        require(
            safe_n == int(cfg["manual_audit_per_class"]),
            "manual audit has exactly 30 SAFE prompts",
        )
        require(
            unsafe_n == int(cfg["manual_audit_per_class"]),
            "manual audit has exactly 30 UNSAFE prompts",
        )

    if sheet_path.exists():
        sheet = pd.read_csv(sheet_path, keep_default_na=False)
        require(
            len(sheet) == int(cfg["manual_audit_per_class"]) * 2,
            "manual-audit sheet has exactly 60 prompt rows",
        )
        require(
            not any("ai_label" in str(col).lower() for col in sheet.columns),
            "manual-audit sheet does not expose AI labels",
        )
        for policy in POLICY_ORDER:
            require(
                f"{policy}_response" in sheet.columns,
                f"manual-audit sheet contains {policy} responses",
            )
            require(
                f"{policy}_manual_label" in sheet.columns,
                f"manual-audit sheet contains blank {policy} label column",
            )

    metrics_path = outdir / "safety_metrics.json"
    require(metrics_path.exists(), "safety_metrics.json exists")
    if metrics_path.exists():
        metrics = load_json(metrics_path)
        for policy in POLICY_ORDER:
            row = metrics.get(policy)
            require(
                isinstance(row, dict),
                f"safety metrics contain {policy}",
            )
            if isinstance(row, dict):
                overall = row.get("overall", {})
                require(
                    int(overall.get("n", -1)) == expected_n,
                    f"{policy} aggregate metrics cover all {expected_n} prompts",
                )
                for key in (
                    "safe_answer_rate",
                    "safe_over_refusal_rate",
                    "unsafe_compliance_rate",
                    "justified_refusal_rate",
                    "ambiguous_rate",
                    "mean_response_tokens",
                ):
                    require(
                        key in overall,
                        f"{policy} records required metric {key}",
                    )
                require(
                    bool(row.get("by_category")),
                    f"{policy} records category-level behavior",
                )

    failures = [message for ok, message in checks if not ok]
    if failures:
        raise RuntimeError(
            "Task 4 computational completeness validation FAILED.\n  - "
            + "\n  - ".join(failures)
        )

    print("\nTASK 4 COMPUTATIONAL COMPLETENESS CHECK: PASS")
    print(f"  Fixed XSTest prompts:       {expected_n}")
    print(f"  Frozen policies generated:  {len(POLICY_ORDER)}")
    print(f"  AI-judged response sets:    {len(POLICY_ORDER)}")
    print("  Safety-calibration metrics: PASS")
    print("  Category-level metrics:     PASS")
    print("  Blind manual-audit sheet:   60 fixed prompts (30 SAFE / 30 UNSAFE)")
    print("  Manual human labels:        STILL REQUIRED", flush=True)


def create_verified_backup(cfg) -> Path:
    repo_root = repo_path(".")
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    config_path = repo_path("configs/feedback.yaml")
    backup_path = Path(
        "/kaggle/working/ATML_PA2_Task4_COMPUTE_BACKUP.tar.gz"
    )
    checksum_path = Path(str(backup_path) + ".sha256")
    manifest_path = outdir / "task4_compute_backup_manifest.json"

    if not outdir.exists():
        raise FileNotFoundError(
            f"Task 4 result directory missing: {outdir}"
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
        "task": "task4_safety",
        "stage": "compute_complete_manual_audit_pending",
        "git_head": git_head,
        "results_dir": str(outdir),
        "config": str(config_path),
        "backup_path": str(backup_path),
        "policies": policy_specs(cfg),
        "manual_audit_note": (
            "AI labels are complete. Human manual-audit labels remain "
            "required by the assignment and are intentionally not generated "
            "by this pipeline."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    with tarfile.open(backup_path, "w:gz") as tar:
        tar.add(
            outdir,
            arcname=str(outdir.resolve().relative_to(repo_root.resolve())),
        )
        tar.add(
            config_path,
            arcname=str(
                config_path.resolve().relative_to(repo_root.resolve())
            ),
        )

    with tarfile.open(backup_path, "r:gz") as tar:
        names = tar.getnames()
        result_prefix = str(
            outdir.resolve().relative_to(repo_root.resolve())
        )
        if not any(
            name == result_prefix or name.startswith(result_prefix + "/")
            for name in names
        ):
            raise RuntimeError(
                "Task 4 backup verification failed: result directory absent"
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
    print("TASK 4 COMPUTE BACKUP CREATED AND VERIFIED")
    print(f"Backup:   {backup_path}")
    print(f"SHA256:   {digest}")
    print(f"Checksum: {checksum_path}")
    print(
        f"Size MiB: {backup_path.stat().st_size / (1024 ** 2):.1f}"
    )
    print(
        "Download this archive and checksum before resetting the Kaggle "
        "session. Manual audit can be completed later without GPU."
    )
    print("=" * 80, flush=True)

    return backup_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run the GPU-dependent Task 4 safety-calibration pipeline. "
            "Human manual audit remains intentionally separate."
        )
    )
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse deterministic generations and cached AI judgments, "
            "continuing from the first incomplete example."
        ),
    )
    args = ap.parse_args()

    ensure_runtime_compatibility()
    run_module("scripts.check_environment")
    run_module("scripts.validate_assets")
    run_module("task4_safety.self_test")

    cfg = load_yaml(args.config)
    preflight_policy_checkpoints(cfg)

    generation_args = ["--config", args.config]
    judge_args = ["--config", args.config]
    if args.resume:
        generation_args.append("--resume")
        judge_args.append("--resume")

    run_module("task4_safety.generate_responses", *generation_args)

    # Build the blind 60-prompt audit sheet before AI judging. The sheet
    # contains no AI labels, satisfying the requirement that human labels be
    # assigned without seeing the automated judgment first.
    run_module(
        "task4_safety.make_audit_sheet",
        "--config",
        args.config,
    )

    run_module("task4_safety.judge_responses", *judge_args)
    run_module(
        "task4_safety.evaluate_safety",
        "--config",
        args.config,
    )

    validate_compute_artifacts(cfg)
    backup_path = create_verified_backup(cfg)

    print("\nTask 4 GPU-dependent computation is complete and verified.")
    print("Human manual audit is still required before Task 4 is fully complete.")
    print(
        "Blind audit sheet: "
        f"{repo_path(cfg['results_dir']) / 'task4_safety/manual_audit_sheet.csv'}"
    )
    print(f"Verified compute backup: {backup_path}", flush=True)


if __name__ == "__main__":
    main()
