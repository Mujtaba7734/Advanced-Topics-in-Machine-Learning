from __future__ import annotations

import argparse
import gc
import hashlib
from collections import defaultdict

import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import JUDGE_PROTOCOL_VERSION, PairwiseAIJudge


EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}

PAIRED_VARIANTS = sorted(EXPECTED_VARIANTS - {"clean_correct"})


def _sha(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        pid = str(row["problem_id"])
        variant = row["variant_type"]
        if variant in by_problem[pid]:
            raise ValueError(f"Duplicate diagnostic variant {variant!r} for problem {pid}")
        by_problem[pid][variant] = row

    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        extra = set(variants) - EXPECTED_VARIANTS
        if missing or extra:
            raise ValueError(
                f"Problem {pid} diagnostic variants mismatch; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
    return by_problem


def _required(row, keys, description):
    for key in keys:
        if key in row and row[key] is not None:
            return str(row[key])
    raise ValueError(
        f"Diagnostic row missing {description}; accepted fields: {keys}"
    )


def _current_pair_metadata(pid, variant_type, variants):
    clean = variants["clean_correct"]
    variant = variants[variant_type]
    clean_response = _required(clean, ("response", "completion"), "response")
    variant_response = _required(variant, ("response", "completion"), "response")
    return {
        "problem_id": str(pid),
        "variant_type": variant_type,
        "clean_sha256": _sha(clean_response),
        "variant_sha256": _sha(variant_response),
    }


def _load_valid_records(path, groups):
    existing = read_jsonl(path) if path.exists() else []
    valid = []
    seen = set()

    for row in existing:
        pid = str(row.get("problem_id"))
        variant = row.get("variant_type")
        key = (pid, variant)
        if (
            pid not in groups
            or variant not in PAIRED_VARIANTS
            or key in seen
            or row.get("preference") not in {"A", "B", "TIE"}
            or int(row.get("judge_protocol_version", 0)) != JUDGE_PROTOCOL_VERSION
        ):
            continue

        expected = _current_pair_metadata(pid, variant, groups[pid])
        if (
            row.get("clean_response_sha256") != expected["clean_sha256"]
            or row.get("variant_response_sha256") != expected["variant_sha256"]
        ):
            continue

        valid.append(row)
        seen.add(key)

    if len(valid) != len(existing):
        write_jsonl(path, valid)
        if existing:
            print(
                f"[resume] Retained {len(valid)}/{len(existing)} diagnostic "
                f"pair judgments matching the current data/protocol.",
                flush=True,
            )
    return valid


def _rates(preferences):
    n = len(preferences)
    if n == 0:
        return {
            "better_response_rate": None,
            "tie_rate": None,
            "wrong_preference_rate": None,
            "pairwise_score": None,
        }
    return {
        "better_response_rate": preferences.count("A") / n,
        "tie_rate": preferences.count("TIE") / n,
        "wrong_preference_rate": preferences.count("B") / n,
        "pairwise_score": (
            preferences.count("A") + 0.5 * preferences.count("TIE")
        ) / n,
    }


def score_groups(cfg, groups):
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "perturbation_pairs.jsonl"

    records = _load_valid_records(path, groups)
    done = {
        (str(r["problem_id"]), r["variant_type"])
        for r in records
    }
    expected = {
        (str(pid), variant)
        for pid in groups
        for variant in PAIRED_VARIANTS
    }
    missing = expected - done

    judge = None
    if missing:
        judge = PairwiseAIJudge(
            cfg,
            outdir / "pairwise_judge_cache.json",
        )

    try:
        for pid, variants in groups.items():
            clean = variants["clean_correct"]
            problem = _required(
                clean,
                ("problem", "question", "prompt"),
                "problem",
            )
            gold = _required(
                clean,
                ("gold_final", "gold_answer"),
                "gold final answer",
            )
            clean_response = _required(
                clean,
                ("response", "completion"),
                "response",
            )

            for variant_type in PAIRED_VARIANTS:
                if (str(pid), variant_type) in done:
                    continue

                variant = variants[variant_type]
                response = _required(
                    variant,
                    ("response", "completion"),
                    "response",
                )
                pref = judge.compare(problem, clean_response, response)
                records.append(
                    {
                        "problem_id": str(pid),
                        "variant_type": variant_type,
                        "preference": pref,
                        "clean_exact_reward": exact_reward(
                            clean_response,
                            gold,
                        ),
                        "variant_exact_reward": exact_reward(
                            response,
                            gold,
                        ),
                        "clean_response": clean_response,
                        "variant_response": response,
                        "clean_response_sha256": _sha(clean_response),
                        "variant_response_sha256": _sha(response),
                        "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
                    }
                )
                done.add((str(pid), variant_type))
                write_jsonl(path, records)
    finally:
        if judge is not None:
            del judge
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if len(records) != len(expected):
        raise RuntimeError(
            f"Expected {len(expected)} controlled diagnostic pairs; "
            f"found {len(records)}"
        )

    categories = {}

    clean_rewards = [
        exact_reward(
            _required(
                variants["clean_correct"],
                ("response", "completion"),
                "response",
            ),
            _required(
                variants["clean_correct"],
                ("gold_final", "gold_answer"),
                "gold final answer",
            ),
        )
        for variants in groups.values()
    ]
    categories["clean_correct"] = {
        "n": len(clean_rewards),
        "role": "reference response; paired perturbations are compared against this baseline",
        "rlvr_reward": sum(clean_rewards) / len(clean_rewards),
    }

    for variant in PAIRED_VARIANTS:
        subset = [
            r for r in records
            if r["variant_type"] == variant
        ]
        n = len(subset)
        if n != len(groups):
            raise RuntimeError(
                f"Expected {len(groups)} pairs for {variant}; found {n}"
            )

        rlvr = [
            "A" if r["clean_exact_reward"] > r["variant_exact_reward"]
            else "B" if r["clean_exact_reward"] < r["variant_exact_reward"]
            else "TIE"
            for r in subset
        ]
        rlaif = [r["preference"] for r in subset]

        categories[variant] = {
            "n": n,
            "rlvr": _rates(rlvr),
            "rlaif": _rates(rlaif),
            "rlvr_clean_reward": (
                sum(r["clean_exact_reward"] for r in subset) / n
            ),
            "rlvr_variant_reward": (
                sum(r["variant_exact_reward"] for r in subset) / n
            ),
        }

    def sensitivity(variant):
        return {
            mechanism: categories[variant][mechanism][
                "better_response_rate"
            ]
            for mechanism in ("rlvr", "rlaif")
        }

    return {
        "n_problems": len(groups),
        "n_responses": sum(len(v) for v in groups.values()),
        "n_pairs": len(records),
        "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
        "category_order": [
            "clean_correct",
            "corrupt_reasoning_correct_final",
            "good_reasoning_wrong_final",
            "persuasive_filler_correct",
            "gold_distractor_wrong_final",
        ],
        "categories": categories,
        "S_reason": sensitivity(
            "corrupt_reasoning_correct_final"
        ),
        "S_outcome": sensitivity(
            "good_reasoning_wrong_final"
        ),
        "sensitivity_definition":
            "Probability that clean/correct-final reward exceeds the matched perturbation reward",
        "filler_susceptibility": {
            mechanism:
                categories["persuasive_filler_correct"][mechanism][
                    "wrong_preference_rate"
                ]
            for mechanism in ("rlvr", "rlaif")
        },
        "distractor_susceptibility": {
            mechanism:
                categories["gold_distractor_wrong_final"][mechanism][
                    "wrong_preference_rate"
                ]
            for mechanism in ("rlvr", "rlaif")
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(
        cfg["paths"]["task5_diagnostics"]
    )
    if len(groups) != 20:
        raise ValueError(
            f"Expected 20 diagnostic problems; found {len(groups)}"
        )
    if sum(len(v) for v in groups.values()) != 100:
        raise ValueError(
            "Expected fixed 100-response perturbation set"
        )
    metrics = score_groups(cfg, groups)
    save_json(
        repo_path(cfg["results_dir"])
        / "task5_feedback"
        / "perturbation_metrics.json",
        metrics,
    )
    print(metrics)


if __name__ == "__main__":
    main()
