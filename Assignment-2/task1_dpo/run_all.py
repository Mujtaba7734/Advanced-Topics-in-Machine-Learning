from __future__ import annotations

import argparse
import subprocess
import sys
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

    print("\nTask 1 pipeline complete.", flush=True)
    print(f"Results: {repo_path(cfg['results_dir'])}", flush=True)
    print(f"Standard adapter: {standard_adapter}", flush=True)
    print(f"Length-balanced adapter: {repo_path(cfg['length_output'])}", flush=True)


if __name__ == "__main__":
    main()
