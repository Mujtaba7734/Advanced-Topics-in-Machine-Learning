from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json


EXPECTED_CACHE_K = 8


def load_k8_cache(path):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("GRPO group-size cache is empty")

    required = {"source_index", "generation_index", "completion", "reward"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(
            f"Unexpected GRPO cache schema; missing fields: {sorted(missing)}"
        )

    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)

    bad = {}
    for pid, group in by_prompt.items():
        indices = [int(row["generation_index"]) for row in group]
        if len(group) < EXPECTED_CACHE_K:
            bad[pid] = f"only {len(group)} completions"
            continue
        if len(indices) != len(set(indices)):
            bad[pid] = "duplicate generation_index values"
            continue

        group.sort(key=lambda row: int(row["generation_index"]))
        first = group[:EXPECTED_CACHE_K]
        first_indices = [int(row["generation_index"]) for row in first]
        if first_indices != list(range(EXPECTED_CACHE_K)):
            bad[pid] = (
                f"first eight generation indices are {first_indices}, "
                f"expected {list(range(EXPECTED_CACHE_K))}"
            )

    if bad:
        preview = dict(list(bad.items())[:10])
        raise ValueError(
            "Invalid fixed K=8 GRPO cache groups; first failures: "
            f"{preview}"
        )

    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Partition the same eight cached completions per prompt into K-sized groups."""
    if k not in (2, 4, 8) or EXPECTED_CACHE_K % k:
        raise ValueError("Expected K in {2, 4, 8}")

    groups = []
    for pid, rows in by_prompt.items():
        fixed = rows[:EXPECTED_CACHE_K]
        for start in range(0, EXPECTED_CACHE_K, k):
            groups.append(
                {
                    "source_index": pid,
                    "generation_indices": [
                        int(row["generation_index"])
                        for row in fixed[start : start + k]
                    ],
                    "rows": fixed[start : start + k],
                }
            )
    return groups


def _difficulty_binning(by_prompt):
    prompt_means = {
        pid: float(
            np.mean(
                [float(row["reward"]) for row in rows[:EXPECTED_CACHE_K]]
            )
        )
        for pid, rows in by_prompt.items()
    }
    values = np.asarray(list(prompt_means.values()), dtype=float)
    thresholds = np.quantile(values, [0.25, 0.50, 0.75])

    labels = [
        "q1_low_reward",
        "q2_mid_low",
        "q3_mid_high",
        "q4_high_reward",
    ]

    def difficulty(pid):
        index = int(
            np.searchsorted(
                thresholds,
                prompt_means[pid],
                side="right",
            )
        )
        return labels[index]

    return prompt_means, thresholds, difficulty


def analyze(cfg):
    by_prompt = load_k8_cache(cfg["group_cache"])
    prompt_means, thresholds, difficulty = _difficulty_binning(by_prompt)

    results = {}
    total_fixed_generations = len(by_prompt) * EXPECTED_CACHE_K

    for k_value in cfg["group_sizes"]:
        k = int(k_value)
        groups = regroup_equal_generation_budget(by_prompt, k)
        records = []

        for group in groups:
            rewards = np.asarray(
                [float(row["reward"]) for row in group["rows"]],
                dtype=float,
            )
            std = float(np.std(rewards))
            relative = (rewards - rewards.mean()) / (std + 1e-6)

            records.append(
                {
                    "source_index": group["source_index"],
                    "generation_indices": group["generation_indices"],
                    "difficulty": difficulty(group["source_index"]),
                    "prompt_mean_reward_k8": prompt_means[
                        group["source_index"]
                    ],
                    "reward_std": std,
                    "informative": bool(std > 1e-6),
                    "relative_signal_variance": float(
                        np.var(relative)
                    ),
                }
            )

        generation_count = sum(len(group["rows"]) for group in groups)
        if generation_count != total_fixed_generations:
            raise RuntimeError(
                f"K={k} used {generation_count} cached generations; "
                f"expected equal budget {total_fixed_generations}"
            )

        by_difficulty = defaultdict(list)
        for record in records:
            by_difficulty[record["difficulty"]].append(record)

        difficulty_summary = {}
        for label, items in sorted(by_difficulty.items()):
            difficulty_summary[label] = {
                "n_groups": len(items),
                "informative_group_fraction": float(
                    np.mean([item["informative"] for item in items])
                ),
                "mean_reward_std": float(
                    np.mean([item["reward_std"] for item in items])
                ),
                "mean_relative_signal_variance": float(
                    np.mean(
                        [
                            item["relative_signal_variance"]
                            for item in items
                        ]
                    )
                ),
            }

        if len(difficulty_summary) < 2:
            raise RuntimeError(
                "Prompt-difficulty analysis produced fewer than two non-empty bins"
            )

        results[str(k)] = {
            "k": k,
            "groups": len(groups),
            "generations": generation_count,
            "informative_group_fraction": float(
                np.mean([item["informative"] for item in records])
            ),
            "mean_reward_std": float(
                np.mean([item["reward_std"] for item in records])
            ),
            "mean_relative_signal_variance": float(
                np.mean(
                    [
                        item["relative_signal_variance"]
                        for item in records
                    ]
                )
            ),
            "by_prompt_difficulty": difficulty_summary,
            "group_records": records,
        }

    generation_budgets = {
        int(key): int(value["generations"])
        for key, value in results.items()
    }
    if len(set(generation_budgets.values())) != 1:
        raise RuntimeError(
            f"Group-size conditions have unequal generation budgets: {generation_budgets}"
        )

    results["metadata"] = {
        "cache_k": EXPECTED_CACHE_K,
        "cached_prompts": len(by_prompt),
        "fixed_generations_per_prompt": EXPECTED_CACHE_K,
        "total_generations_per_condition": total_fixed_generations,
        "group_sizes": [int(k) for k in cfg["group_sizes"]],
        "difficulty_rule": (
            "Quartiles of each prompt's mean learned reward over the same fixed K=8 "
            "cached completions; q1 has the lowest mean reward and q4 the highest."
        ),
        "difficulty_quantile_thresholds": [
            float(value) for value in thresholds
        ],
    }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    results = analyze(cfg)
    out = repo_path(cfg["results_dir"]) / "group_size_study.json"
    save_json(out, results)

    print(
        "Task 3 group-size preflight/study: PASS "
        f"({results['metadata']['cached_prompts']} prompts, "
        f"{results['metadata']['total_generations_per_condition']} fixed "
        "generations per K condition)",
        flush=True,
    )


if __name__ == "__main__":
    main()
