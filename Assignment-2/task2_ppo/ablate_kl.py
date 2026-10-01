from __future__ import annotations

import argparse
from pathlib import Path
from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import load_evaluation_bundle
from common.rl_evaluation import evaluate_rl


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


def completed_training(cfg, name: str, adapter: Path, expected_updates: int, expected_beta: float):
    value_adapter = adapter.parent / f"{adapter.name}_value"
    summary_path = repo_path(cfg["results_dir"]) / f"{name}_train.json"
    log_path = repo_path(cfg["results_dir"]) / f"{name}_train.jsonl"
    if not (adapter_is_complete(adapter) and adapter_is_complete(value_adapter)
            and summary_path.exists() and jsonl_count(log_path) == expected_updates):
        return None
    summary = load_json(summary_path)
    if int(summary.get("updates", -1)) != expected_updates:
        return None
    if abs(float(summary.get("kl_beta", -999.0)) - float(expected_beta)) > 1e-12:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse completed PPO KL forks/evaluations.")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    expected_eval_n = len(read_jsonl(cfg["paths"]["rl_prompt_eval"]))
    study_path = repo_path(cfg["results_dir"]) / "kl_study.json"
    results = []
    for beta in cfg["kl_values"]:
        name = f"kl_{float(beta):.2f}"
        adapter = repo_path(cfg["output"]).parent / name
        train = completed_training(
            cfg, name, adapter, int(cfg["fork_updates"]), float(beta)
        ) if args.resume else None
        if train is not None:
            print(f"[resume] Reusing completed training: {name}", flush=True)
        else:
            train = run_ppo(
                args.config, str(adapter), int(cfg["fork_updates"]),
                kl_beta=float(beta), run_name=name
            )

        evaluation = completed_evaluation(cfg, name, expected_eval_n) if args.resume else None
        if evaluation is not None:
            print(f"[resume] Reusing completed evaluation: {name}", flush=True)
        else:
            evaluation = evaluate_rl(
                load_evaluation_bundle(args.config, str(adapter)), name, "ppo"
            )

        results.append({
            "kl_beta": float(beta),
            "train": train,
            "evaluation": evaluation,
        })
        save_json(study_path, results)


if __name__ == "__main__":
    main()
