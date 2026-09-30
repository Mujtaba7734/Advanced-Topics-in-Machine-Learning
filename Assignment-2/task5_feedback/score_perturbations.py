from __future__ import annotations

import argparse
from collections import defaultdict

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def score_groups(cfg, groups):
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    judge = PairwiseAIJudge(cfg, outdir / "pairwise_judge_cache.json")
    path = outdir / "perturbation_pairs.jsonl"
    records = read_jsonl(path) if path.exists() else []
    done = {(str(r["problem_id"]), r["variant_type"]) for r in records}
    def required(row, keys, description):
        for key in keys:
            if key in row and row[key] is not None:
                return str(row[key])
        raise ValueError(f"Diagnostic row missing {description}; accepted fields: {keys}")
    for pid, variants in groups.items():
        clean = variants["clean_correct"]
        problem = required(clean, ("problem", "question", "prompt"), "problem")
        gold = required(clean, ("gold_final", "gold_answer"), "gold final answer")
        clean_response = required(clean, ("response", "completion"), "response")
        for variant_type, variant in variants.items():
            if variant_type == "clean_correct" or (pid, variant_type) in done:
                continue
            response = required(variant, ("response", "completion"), "response")
            pref = judge.compare(problem, clean_response, response)
            records.append({"problem_id": pid, "variant_type": variant_type,
                "preference": pref, "clean_exact_reward": exact_reward(clean_response, gold),
                "variant_exact_reward": exact_reward(response, gold),
                "clean_response": clean_response, "variant_response": response})
            write_jsonl(path, records)
    categories = {}
    clean_rewards = [exact_reward(required(v["clean_correct"], ("response", "completion"), "response"),
                                  required(v["clean_correct"], ("gold_final", "gold_answer"), "gold final answer"))
                     for v in groups.values()]
    categories["clean_correct"] = {"n": len(clean_rewards),
                                   "rlvr_reward": sum(clean_rewards) / len(clean_rewards)}
    for variant in sorted(EXPECTED_VARIANTS - {"clean_correct"}):
        subset = [r for r in records if r["variant_type"] == variant]
        n = len(subset)
        rlvr = ["A" if r["clean_exact_reward"] > r["variant_exact_reward"] else
                "B" if r["clean_exact_reward"] < r["variant_exact_reward"] else "TIE"
                for r in subset]
        rlaif = [r["preference"] for r in subset]
        def rates(preferences):
            return {"better_response_rate": preferences.count("A") / n,
                    "tie_rate": preferences.count("TIE") / n,
                    "wrong_preference_rate": preferences.count("B") / n,
                    "pairwise_score": (preferences.count("A") + 0.5 * preferences.count("TIE")) / n}
        categories[variant] = {"n": n, "rlvr": rates(rlvr), "rlaif": rates(rlaif),
            "rlvr_clean_reward": sum(r["clean_exact_reward"] for r in subset) / n,
            "rlvr_variant_reward": sum(r["variant_exact_reward"] for r in subset) / n}
    def sensitivity(variant):
        return {mechanism: categories[variant][mechanism]["better_response_rate"]
                for mechanism in ("rlvr", "rlaif")}
    return {"n_problems": len(groups), "n_responses": sum(len(v) for v in groups.values()),
        "n_pairs": len(records), "categories": categories,
        "S_reason": sensitivity("corrupt_reasoning_correct_final"),
        "S_outcome": sensitivity("good_reasoning_wrong_final"),
        "sensitivity_definition": "Probability that clean/correct-final reward exceeds the matched perturbation reward",
        "filler_susceptibility": {mechanism: categories["persuasive_filler_correct"][mechanism]["wrong_preference_rate"]
                                  for mechanism in ("rlvr", "rlaif")},
        "distractor_susceptibility": {mechanism: categories["gold_distractor_wrong_final"][mechanism]["wrong_preference_rate"]
                                      for mechanism in ("rlvr", "rlaif")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    if sum(len(v) for v in groups.values()) != 100:
        raise ValueError("Expected fixed 100-response perturbation set")
    metrics = score_groups(cfg, groups)
    save_json(repo_path(cfg["results_dir"]) / "task5_feedback" / "perturbation_metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
