from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common.data import load_yaml, repo_path


def run_module(module: str, *args: str) -> None:
    command = [sys.executable, "-m", module, *args]
    print("\n" + "=" * 80)
    print("Running:", " ".join(command))
    print("=" * 80, flush=True)
    subprocess.run(command, check=True)


def standard_adapter_is_complete(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "adapter_config.json").exists()
    has_weights = any(
        (path / name).exists()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )
    return has_config and has_weights


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the complete Task 1 DPO experiment pipeline."
    )
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument(
        "--skip-standard-train",
        action="store_true",
        help="Reuse an already completed standard DPO adapter instead of retraining it.",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    standard_adapter = repo_path(cfg["standard_output"])

    if args.skip_standard_train:
        if not standard_adapter_is_complete(standard_adapter):
            raise FileNotFoundError(
                f"--skip-standard-train was requested, but no complete adapter exists at {standard_adapter}"
            )
        print(f"Reusing completed standard DPO adapter: {standard_adapter}", flush=True)
    else:
        run_module(
            "task1_dpo.train",
            "--config", args.config,
            "--run-name", "standard",
            "--output", str(standard_adapter),
        )

    run_module(
        "task1_dpo.evaluate",
        "--config", args.config,
        "--adapter", str(standard_adapter),
        "--name", "standard",
    )

    run_module(
        "task1_dpo.ablate_beta",
        "--config", args.config,
    )

    run_module(
        "task1_dpo.analyze_length",
        "--config", args.config,
    )

    print("\nTask 1 pipeline complete.", flush=True)
    print(f"Results: {repo_path(cfg['results_dir'])}", flush=True)
    print(f"Standard adapter: {standard_adapter}", flush=True)
    print(f"Length-balanced adapter: {repo_path(cfg['length_output'])}", flush=True)


if __name__ == "__main__":
    main()
