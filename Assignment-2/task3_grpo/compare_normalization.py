from __future__ import annotations

import argparse
from common.data import load_yaml, repo_path, read_jsonl
from common.logging_utils import save_json
from common.rl_evaluation import evaluate_rl
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import load_evaluation_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = {}
    examples = {}
    optimization = {}
    for loss_type in ("grpo", "dr_grpo"):
        name = f"normalization_{loss_type}"
        adapter = repo_path(cfg["output"]).parent / name
        train = run_grpo(args.config, str(adapter), int(cfg["fork_updates"]), loss_type, name,
                         log_optimization_details=True)
        evaluation = evaluate_rl(load_evaluation_bundle(args.config, str(adapter)), name, "grpo")
        examples[loss_type] = read_jsonl(repo_path(cfg["results_dir"]) / f"{name}_examples.jsonl")
        optimization[loss_type] = read_jsonl(repo_path(cfg["results_dir"]) / f"{name}_optimization.jsonl")
        results[loss_type] = {"train": train, "evaluation": evaluation}
    lengths = sorted(r["response_tokens"] for rows in examples.values() for r in rows)
    median = lengths[len(lengths) // 2] if lengths else None
    for loss_type, rows in examples.items():
        conditioned = {}
        for label, subset in (("short", [r for r in rows if median is not None and r["response_tokens"] <= median]),
                              ("long", [r for r in rows if median is not None and r["response_tokens"] > median])):
            conditioned[label] = {"n": len(subset), "mean_reward":
                sum(r["reward"] for r in subset) / len(subset) if subset else None,
                "mean_response_tokens": sum(r["response_tokens"] for r in subset) / len(subset) if subset else None}
        results[loss_type]["length_conditioned"] = conditioned
        results[loss_type]["shared_median_length"] = median
    training_lengths = sorted(r["response_tokens"] for rows in optimization.values() for r in rows)
    training_median = training_lengths[len(training_lengths) // 2] if training_lengths else None
    for loss_type, rows in optimization.items():
        results[loss_type]["training_length_threshold"] = training_median
        results[loss_type]["training_length_conditioned"] = {}
        for label, subset in (("short", [r for r in rows if training_median is not None and r["response_tokens"] <= training_median]),
                              ("long", [r for r in rows if training_median is not None and r["response_tokens"] > training_median])):
            results[loss_type]["training_length_conditioned"][label] = {
                "n": len(subset),
                "mean_normalization_weight_per_token":
                    sum(r["normalization_weight_per_token"] for r in subset) / len(subset) if subset else None,
                "mean_absolute_policy_loss_contribution":
                    sum(r["absolute_policy_loss_contribution"] for r in subset) / len(subset) if subset else None,
                "mean_normalized_clipped_surrogate":
                    sum(r["normalized_clipped_surrogate"] for r in subset) / len(subset) if subset else None,
                "masked_fraction": sum(r["optimization_tokens"] == 0 for r in subset) / len(subset) if subset else None,
            }
    save_json(repo_path(cfg["results_dir"]) / "normalization_study.json", results)


if __name__ == "__main__":
    main()
