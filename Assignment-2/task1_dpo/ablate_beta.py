from __future__ import annotations

import argparse
from pathlib import Path

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from task1_dpo.train import run_training
from task1_dpo.evaluate import run_evaluation


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


def completed_evaluation(cfg, name: str, expected_n: int):
    metrics_path = repo_path(cfg["results_dir"]) / f"{name}_eval.json"
    examples_path = repo_path(cfg["results_dir"]) / f"{name}_examples.jsonl"
    if not metrics_path.exists() or jsonl_count(examples_path) != expected_n:
        return None
    metrics = load_json(metrics_path)
    if int(metrics.get("n", -1)) != expected_n:
        return None
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse completed beta adapters/evaluations and continue from the first incomplete stage.")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    expected_eval_n = len(read_jsonl(cfg["paths"]["dpo_standard_eval"]))
    results = []

    for beta in cfg["betas"]:
        name = f"beta_{float(beta):.2f}"
        adapter = repo_path(cfg["standard_output"]).parent / name
        train_path = repo_path(cfg["results_dir"]) / f"{name}_train.json"

        if args.resume and adapter_is_complete(adapter) and train_path.exists():
            train = load_json(train_path)
            print(f"[resume] Reusing completed training: {name}", flush=True)
        else:
            train = run_training(
                args.config,
                name,
                output_path=str(adapter),
                beta=float(beta),
                max_examples=int(cfg["short_ablation_examples"]),
            )

        evaluation = completed_evaluation(cfg, name, expected_eval_n) if args.resume else None
        if evaluation is not None:
            print(f"[resume] Reusing completed evaluation: {name}", flush=True)
        else:
            evaluation = run_evaluation(args.config, str(adapter), name, beta=float(beta), resume=args.resume)

        results.append({"beta": float(beta), "train": train, "evaluation": evaluation})
        # Save after every condition so an interruption never loses completed aggregate work.
        save_json(repo_path(cfg["results_dir"]) / "beta_study.json", results)


if __name__ == "__main__":
    main()
