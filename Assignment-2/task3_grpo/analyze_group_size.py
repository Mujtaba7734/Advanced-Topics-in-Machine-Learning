from __future__ import annotations

import argparse
from collections import defaultdict
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Use the first eight indexed completions after a deterministic sort.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Return K-sized groups while keeping total cached completions fixed.

    Students should decide and document exactly how prompts/completions are partitioned for the
    requested equal-generation comparison.
    """
    if k not in (2, 4, 8) or 8 % k:
        raise ValueError("Expected K in {2,4,8}")
    groups = []
    for pid, rows in by_prompt.items():
        fixed = rows[:8]
        if len(fixed) != 8:
            raise ValueError(f"Prompt {pid} has fewer than eight cached completions")
        for start in range(0, 8, k):
            groups.append({"source_index": pid, "rows": fixed[start:start + k]})
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])
    results = {}
    prompt_means = {pid: np.mean([float(r["reward"]) for r in rows[:8]]) for pid, rows in by_prompt.items()}
    thresholds = np.quantile(list(prompt_means.values()), [0.25, 0.5, 0.75])
    def difficulty(pid):
        return ["q1_low_reward", "q2", "q3", "q4_high_reward"][int(np.searchsorted(thresholds, prompt_means[pid], side="right"))]
    for k in cfg["group_sizes"]:
        groups = regroup_equal_generation_budget(by_prompt, int(k))
        records = []
        for group in groups:
            rewards = np.asarray([float(r["reward"]) for r in group["rows"]])
            std = float(np.std(rewards))
            relative = (rewards - rewards.mean()) / (std + 1e-6)
            records.append({"source_index": group["source_index"], "difficulty": difficulty(group["source_index"]),
                "reward_std": std, "informative": std > 1e-6,
                "relative_signal_variance": float(np.var(relative))})
        by_difficulty = defaultdict(list)
        for record in records:
            by_difficulty[record["difficulty"]].append(record)
        results[str(k)] = {"groups": len(groups), "generations": sum(len(g["rows"]) for g in groups),
            "informative_group_fraction": float(np.mean([x["informative"] for x in records])),
            "mean_reward_std": float(np.mean([x["reward_std"] for x in records])),
            "mean_relative_signal_variance": float(np.mean([x["relative_signal_variance"] for x in records])),
            "by_prompt_difficulty": {key: {"n": len(items),
                "informative_group_fraction": float(np.mean([x["informative"] for x in items])),
                "mean_reward_std": float(np.mean([x["reward_std"] for x in items]))}
                for key, items in by_difficulty.items()}}
    save_json(repo_path(cfg["results_dir"]) / "group_size_study.json", results)


if __name__ == "__main__":
    main()
