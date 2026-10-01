from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json


def run_module(module: str, *args: str) -> None:
    command = [sys.executable, "-m", module, *args]
    print("\n" + "=" * 80)
    print("Running:", " ".join(command))
    print("=" * 80, flush=True)
    subprocess.run(command, check=True)


def adapter_is_complete(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "adapter_config.json").exists()
    has_weights = any((path / name).exists() for name in ("adapter_model.safetensors", "adapter_model.bin"))
    return has_config and has_weights


def jsonl_count(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def standard_evaluation_is_complete(cfg) -> bool:
    expected_n = len(read_jsonl(cfg["paths"]["dpo_standard_eval"]))
    metrics_path = repo_path(cfg["results_dir"]) / "standard_eval.json"
    examples_path = repo_path(cfg["results_dir"]) / "standard_examples.jsonl"
    if not metrics_path.exists() or jsonl_count(examples_path) != expected_n:
        return False
    metrics = load_json(metrics_path)
    return int(metrics.get("n", -1)) == expected_n




def _jsonl_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def validate_task1_artifacts(cfg) -> None:
    """Fail loudly if any required Task 1 run was shortened or is incomplete."""
    results_dir = repo_path(cfg["results_dir"])
    standard_train_rows = len(read_jsonl(cfg["paths"]["dpo_standard_train"]))
    standard_eval_rows = len(read_jsonl(cfg["paths"]["dpo_standard_eval"]))
    length_train_rows = len(read_jsonl(cfg["paths"]["dpo_length_train"]))
    length_eval_rows = len(read_jsonl(cfg["paths"]["dpo_length_eval"]))
    word_limit_rows = len(read_jsonl(cfg["paths"]["word_limit_prompts"]))

    checks = []

    def require(condition, message):
        checks.append((bool(condition), message))

    # Standard DPO: full fixed training set, exactly one epoch, full held-out evaluation.
    standard_train_path = results_dir / "standard_train.json"
    standard_eval_path = results_dir / "standard_eval.json"
    standard_examples_path = results_dir / "standard_examples.jsonl"
    require(adapter_is_complete(repo_path(cfg["standard_output"])), "standard DPO adapter exists")
    require(standard_train_path.exists(), "standard_train.json exists")
    if standard_train_path.exists():
        standard_train = load_json(standard_train_path)
        require(int(standard_train.get("examples", -1)) == standard_train_rows,
                f"standard DPO used all {standard_train_rows} fixed training examples")
        require(int(standard_train.get("epochs", -1)) == 1,
                "standard DPO ran exactly one epoch")
    require(standard_eval_path.exists(), "standard_eval.json exists")
    require(standard_examples_path.exists(), "standard_examples.jsonl exists")
    if standard_eval_path.exists():
        require(int(load_json(standard_eval_path).get("n", -1)) == standard_eval_rows,
                f"standard DPO evaluated all {standard_eval_rows} fixed held-out pairs")
    if standard_examples_path.exists():
        require(_jsonl_count(standard_examples_path) == standard_eval_rows,
                f"standard example log contains all {standard_eval_rows} held-out pairs")

    # Beta study: all three required betas, exact short-run budget, full common evaluation.
    beta_study_path = results_dir / "beta_study.json"
    require(beta_study_path.exists(), "beta_study.json exists")
    for beta in cfg["betas"]:
        name = f"beta_{float(beta):.2f}"
        adapter = repo_path(cfg["standard_output"]).parent / name
        train_path = results_dir / f"{name}_train.json"
        eval_path = results_dir / f"{name}_eval.json"
        examples_path = results_dir / f"{name}_examples.jsonl"
        require(adapter_is_complete(adapter), f"{name} adapter exists")
        require(train_path.exists(), f"{name}_train.json exists")
        if train_path.exists():
            train = load_json(train_path)
            require(int(train.get("examples", -1)) == int(cfg["short_ablation_examples"]),
                    f"{name} used exactly {int(cfg['short_ablation_examples'])} short-run examples")
            require(abs(float(train.get("beta", -999.0)) - float(beta)) < 1e-12,
                    f"{name} used beta={float(beta):.2f}")
        require(eval_path.exists(), f"{name}_eval.json exists")
        if eval_path.exists():
            require(int(load_json(eval_path).get("n", -1)) == standard_eval_rows,
                    f"{name} evaluated all {standard_eval_rows} common held-out pairs")
        require(examples_path.exists() and _jsonl_count(examples_path) == standard_eval_rows,
                f"{name} example log contains all {standard_eval_rows} held-out pairs")

    # Length-confounding study: full supplied balanced set and full stratified evaluation for both policies.
    length_train_path = results_dir / "length_balanced_train.json"
    require(adapter_is_complete(repo_path(cfg["length_output"])), "length-balanced adapter exists")
    require(length_train_path.exists(), "length_balanced_train.json exists")
    if length_train_path.exists():
        length_train = load_json(length_train_path)
        require(int(length_train.get("examples", -1)) == length_train_rows,
                f"length-balanced DPO used all {length_train_rows} supplied training pairs")
        require(int(length_train.get("epochs", -1)) == 1,
                "length-balanced DPO ran exactly one epoch")
    for name in ("standard_stratified", "length_balanced_stratified"):
        eval_path = results_dir / f"{name}_eval.json"
        examples_path = results_dir / f"{name}_examples.jsonl"
        require(eval_path.exists(), f"{name}_eval.json exists")
        if eval_path.exists():
            require(int(load_json(eval_path).get("n", -1)) == length_eval_rows,
                    f"{name} evaluated all {length_eval_rows} stratified held-out pairs")
        require(examples_path.exists() and _jsonl_count(examples_path) == length_eval_rows,
                f"{name} example log contains all {length_eval_rows} stratified pairs")

    length_study_path = results_dir / "length_study.json"
    require(length_study_path.exists(), "length_study.json exists")
    for name in ("standard", "length_balanced"):
        word_path = results_dir / f"{name}_word_limits.jsonl"
        require(word_path.exists() and _jsonl_count(word_path) == word_limit_rows,
                f"{name} word-limit evaluation contains all {word_limit_rows} common prompts")

    failures = [message for ok, message in checks if not ok]
    if failures:
        formatted = "\n  - ".join(failures)
        raise RuntimeError(
            "Task 1 completeness validation FAILED. Nothing will be declared complete or backed up.\n"
            f"  - {formatted}"
        )

    print("\nTASK 1 COMPLETENESS CHECK: PASS")
    print(f"  Standard train:       {standard_train_rows}/{standard_train_rows} examples, 1 epoch")
    print(f"  Standard eval:        {standard_eval_rows}/{standard_eval_rows} pairs")
    print(f"  Beta short runs:      {len(cfg['betas'])} x {int(cfg['short_ablation_examples'])} examples")
    print(f"  Beta evals:           {len(cfg['betas'])} x {standard_eval_rows} pairs")
    print(f"  Length-balanced train:{length_train_rows}/{length_train_rows} pairs, 1 epoch")
    print(f"  Stratified evals:     2 x {length_eval_rows} pairs")
    print(f"  Word-limit evals:     2 x {word_limit_rows} prompts")
    print("  No Task 1 experimental budget was shortened.", flush=True)


def create_verified_backup(cfg) -> Path:
    """Archive every irreplaceable Task 1 artifact and verify the archive."""
    repo_root = repo_path(".")
    backup_path = Path("/kaggle/working/ATML_PA2_Task1_BACKUP.tar.gz")
    checksum_path = Path(str(backup_path) + ".sha256")
    manifest_path = repo_path(cfg["results_dir"]) / "task1_backup_manifest.json"

    required = [
        repo_path(cfg["results_dir"]),
        repo_path(cfg["standard_output"]),
        repo_path(cfg["length_output"]),
        repo_path("configs/dpo.yaml"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Refusing to declare Task 1 complete because backup inputs are missing: "
            + ", ".join(missing)
        )

    beta_adapters = [
        repo_path(cfg["standard_output"]).parent / f"beta_{float(beta):.2f}"
        for beta in cfg["betas"]
    ]
    missing_beta = [str(path) for path in beta_adapters if not adapter_is_complete(path)]
    if missing_beta:
        raise FileNotFoundError(
            "Refusing to declare Task 1 complete because beta adapters are missing: "
            + ", ".join(missing_beta)
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
        "task": "task1_dpo",
        "git_head": git_head,
        "results_dir": str(repo_path(cfg["results_dir"])),
        "standard_adapter": str(repo_path(cfg["standard_output"])),
        "length_balanced_adapter": str(repo_path(cfg["length_output"])),
        "beta_adapters": [str(path) for path in beta_adapters],
        "backup_path": str(backup_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    archive_inputs = [
        repo_path(cfg["results_dir"]),
        repo_path(cfg["standard_output"]),
        repo_path(cfg["length_output"]),
        *beta_adapters,
        repo_path("configs/dpo.yaml"),
        manifest_path,
    ]

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

    # Verify that the archive is readable and contains both results and adapters.
    with tarfile.open(backup_path, "r:gz") as tar:
        names = tar.getnames()
        required_prefixes = [
            str(Path(cfg["results_dir"])),
            str(Path(cfg["standard_output"])),
            str(Path(cfg["length_output"])),
        ]
        for prefix in required_prefixes:
            if not any(name == prefix or name.startswith(prefix + "/") for name in names):
                raise RuntimeError(f"Backup verification failed: missing {prefix}")

    sha256 = hashlib.sha256()
    with backup_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
    digest = sha256.hexdigest()
    checksum_path.write_text(f"{digest}  {backup_path.name}\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print("TASK 1 BACKUP CREATED AND VERIFIED")
    print(f"Backup:   {backup_path}")
    print(f"SHA256:   {digest}")
    print(f"Checksum: {checksum_path}")
    print(f"Size MiB: {backup_path.stat().st_size / (1024 ** 2):.1f}")
    print("IMPORTANT: preserve this /kaggle/working backup with Kaggle Save Version/output")
    print("or download it before stopping/resetting the session.")
    print("=" * 80, flush=True)
    return backup_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the complete Task 1 DPO experiment pipeline.")
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument(
        "--skip-standard-train",
        action="store_true",
        help="Reuse an already completed standard DPO adapter instead of retraining it.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Reuse every completed Task 1 stage and continue from the first incomplete stage.",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    standard_adapter = repo_path(cfg["standard_output"])

    if args.skip_standard_train or (args.resume and adapter_is_complete(standard_adapter)):
        if not adapter_is_complete(standard_adapter):
            raise FileNotFoundError(
                f"Requested reuse of standard DPO, but no complete adapter exists at {standard_adapter}"
            )
        print(f"[resume] Reusing completed standard DPO adapter: {standard_adapter}", flush=True)
    else:
        run_module(
            "task1_dpo.train",
            "--config", args.config,
            "--run-name", "standard",
            "--output", str(standard_adapter),
        )

    if args.resume and standard_evaluation_is_complete(cfg):
        print("[resume] Reusing completed standard evaluation", flush=True)
    else:
        standard_eval_args = [
            "--config", args.config,
            "--adapter", str(standard_adapter),
            "--name", "standard",
        ]
        if args.resume:
            standard_eval_args.append("--resume")
        run_module("task1_dpo.evaluate", *standard_eval_args)

    beta_args = ["--config", args.config]
    if args.resume:
        beta_args.append("--resume")
    run_module("task1_dpo.ablate_beta", *beta_args)

    length_args = ["--config", args.config]
    if args.resume:
        length_args.append("--resume")
    run_module("task1_dpo.analyze_length", *length_args)

    validate_task1_artifacts(cfg)
    backup_path = create_verified_backup(cfg)

    print("\nTask 1 pipeline complete and verified.", flush=True)
    print(f"Results: {repo_path(cfg['results_dir'])}", flush=True)
    print(f"Standard adapter: {standard_adapter}", flush=True)
    print(f"Length-balanced adapter: {repo_path(cfg['length_output'])}", flush=True)
    print(f"Verified backup: {backup_path}", flush=True)


if __name__ == "__main__":
    main()
