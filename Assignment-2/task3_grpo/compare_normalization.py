from __future__ import annotations

import argparse
from pathlib import Path

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.rl_evaluation import evaluate_rl
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import load_evaluation_bundle


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


def completed_training(cfg, name: str, adapter: Path, loss_type: str):
    results_dir = repo_path(cfg["results_dir"])
    summary_path = results_dir / f"{name}_train.json"
    train_log = results_dir / f"{name}_train.jsonl"
    details_log = results_dir / f"{name}_optimization.jsonl"

    expected_updates = int(cfg["fork_updates"])
    expected_details = (
        expected_updates
        * int(cfg["prompts_per_update"])
        * int(cfg["num_generations"])
        * int(cfg["policy_epochs"])
    )

    if not (
        adapter_is_complete(adapter)
        and summary_path.exists()
        and jsonl_count(train_log) == expected_updates
        and jsonl_count(details_log) == expected_details
    ):
        return None

    summary = load_json(summary_path)
    checks = [
        int(summary.get("updates", -1)) == expected_updates,
        int(summary.get("num_generations", -1)) == int(cfg["num_generations"]),
        summary.get("loss_type") == loss_type,
        summary.get("midpoint") == cfg["paths"]["grpo_midpoint_policy"],
        abs(
            float(summary.get("clip_epsilon", -999.0))
            - float(cfg["clip_epsilon"])
        )
        < 1e-12,
        abs(float(summary.get("kl_beta", -999.0)) - float(cfg["kl_beta"]))
        < 1e-12,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete normalization stages and resume partial evaluations.",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    results = {}
    examples = {}
    optimization = {}
    expected_eval_n = len(read_jsonl(cfg["paths"]["rl_prompt_eval"]))

    for loss_type in ("grpo", "dr_grpo"):
        name = f"normalization_{loss_type}"
        adapter = repo_path(cfg["output"]).parent / name

        train = (
            completed_training(cfg, name, adapter, loss_type)
            if args.resume
            else None
        )
        if train is not None:
            print(f"[resume] Reusing completed training: {name}", flush=True)
        else:
            train = run_grpo(
                args.config,
                str(adapter),
                int(cfg["fork_updates"]),
                loss_type,
                name,
                log_optimization_details=True,
            )

        evaluation = (
            completed_evaluation(cfg, name, expected_eval_n)
            if args.resume
            else None
        )
        if evaluation is not None:
            print(f"[resume] Reusing completed evaluation: {name}", flush=True)
        else:
            evaluation = evaluate_rl(
                load_evaluation_bundle(args.config, str(adapter)),
                name,
                "grpo",
                resume=args.resume,
            )

        examples[loss_type] = read_jsonl(
            repo_path(cfg["results_dir"]) / f"{name}_examples.jsonl"
        )
        optimization[loss_type] = read_jsonl(
            repo_path(cfg["results_dir"]) / f"{name}_optimization.jsonl"
        )
        results[loss_type] = {
            "train": train,
            "evaluation": evaluation,
        }

    lengths = sorted(
        row["response_tokens"]
        for rows in examples.values()
        for row in rows
    )
    median = lengths[len(lengths) // 2] if lengths else None

    for loss_type, rows in examples.items():
        conditioned = {}
        for label, subset in (
            (
                "short",
                [
                    row
                    for row in rows
                    if median is not None
                    and row["response_tokens"] <= median
                ],
            ),
            (
                "long",
                [
                    row
                    for row in rows
                    if median is not None
                    and row["response_tokens"] > median
                ],
            ),
        ):
            conditioned[label] = {
                "n": len(subset),
                "mean_reward": (
                    sum(row["reward"] for row in subset) / len(subset)
                    if subset
                    else None
                ),
                "mean_sampled_kl": (
                    sum(row["sampled_kl"] for row in subset) / len(subset)
                    if subset
                    else None
                ),
                "mean_response_tokens": (
                    sum(row["response_tokens"] for row in subset) / len(subset)
                    if subset
                    else None
                ),
            }

        results[loss_type]["length_conditioned"] = conditioned
        results[loss_type]["shared_median_length"] = median

    training_lengths = sorted(
        row["response_tokens"]
        for rows in optimization.values()
        for row in rows
    )
    training_median = (
        training_lengths[len(training_lengths) // 2]
        if training_lengths
        else None
    )

    for loss_type, rows in optimization.items():
        results[loss_type]["training_length_threshold"] = training_median
        results[loss_type]["training_length_conditioned"] = {}

        for label, subset in (
            (
                "short",
                [
                    row
                    for row in rows
                    if training_median is not None
                    and row["response_tokens"] <= training_median
                ],
            ),
            (
                "long",
                [
                    row
                    for row in rows
                    if training_median is not None
                    and row["response_tokens"] > training_median
                ],
            ),
        ):
            results[loss_type]["training_length_conditioned"][label] = {
                "n": len(subset),
                "mean_normalization_weight_per_token": (
                    sum(
                        row["normalization_weight_per_token"]
                        for row in subset
                    )
                    / len(subset)
                    if subset
                    else None
                ),
                "mean_absolute_policy_loss_contribution": (
                    sum(
                        row["absolute_policy_loss_contribution"]
                        for row in subset
                    )
                    / len(subset)
                    if subset
                    else None
                ),
                "mean_normalized_clipped_surrogate": (
                    sum(
                        row["normalized_clipped_surrogate"]
                        for row in subset
                    )
                    / len(subset)
                    if subset
                    else None
                ),
                "mean_abs_advantage": (
                    sum(abs(row["advantage"]) for row in subset)
                    / len(subset)
                    if subset
                    else None
                ),
                "masked_fraction": (
                    sum(
                        row["optimization_tokens"] == 0
                        for row in subset
                    )
                    / len(subset)
                    if subset
                    else None
                ),
            }

    budgets = {
        key: int(value["train"]["max_generated_token_budget"])
        for key, value in results.items()
    }
    if len(set(budgets.values())) != 1:
        raise RuntimeError(
            "Normalization forks do not have equal generated-token budgets: "
            f"{budgets}"
        )

    results["comparison_metadata"] = {
        "loss_types": ["grpo", "dr_grpo"],
        "same_midpoint": cfg["paths"]["grpo_midpoint_policy"],
        "same_clip_epsilon": float(cfg["clip_epsilon"]),
        "same_kl_beta": float(cfg["kl_beta"]),
        "same_num_generations": int(cfg["num_generations"]),
        "same_updates": int(cfg["fork_updates"]),
        "max_generated_token_budget_per_fork": next(
            iter(budgets.values())
        ),
        "shared_eval_median_length": median,
        "shared_training_median_length": training_median,
    }

    save_json(
        repo_path(cfg["results_dir"]) / "normalization_study.json",
        results,
    )
    print("Task 3 normalization study: PASS", flush=True)


if __name__ == "__main__":
    main()
