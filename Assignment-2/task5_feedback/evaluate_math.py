from __future__ import annotations

import argparse
import gc
import hashlib
from itertools import combinations

import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed
from common.models import load_policy, load_tokenizer
from task5_feedback.rlaif import JUDGE_PROTOCOL_VERSION, PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def _source_id(row: dict, position: int) -> str:
    return str(row.get("source_index", row.get("prompt_id", position)))


def _response_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _validate_response_prefix(records: list[dict], rows: list[dict], policy: str) -> None:
    if len(records) > len(rows):
        raise ValueError(
            f"Saved {policy} responses contain {len(records)} rows but the fixed set has only {len(rows)}"
        )
    for i, record in enumerate(records):
        expected = _source_id(rows[i], i)
        if str(record.get("source_index")) != expected:
            raise ValueError(
                f"Saved {policy} responses are not a prefix of the fixed evaluation set at row {i}: "
                f"{record.get('source_index')!r} != {expected!r}"
            )
        if record.get("policy") != policy:
            raise ValueError(
                f"Saved response row {i} says policy={record.get('policy')!r}; expected {policy!r}"
            )


def _generate_policy_responses(cfg, rows, tok, dataset: str, policy: str, path):
    batch_size = max(1, int(cfg.get("math_eval_batch_size", 1)))
    seed_by_batch = bool(cfg.get("math_eval_seed_by_batch", False))
    records = read_jsonl(path) if path.exists() else []
    _validate_response_prefix(records, rows, policy)

    if len(records) == len(rows):
        print(f"[resume] Reusing complete {dataset} responses: {policy} ({len(rows)}/{len(rows)})", flush=True)
        return records

    if records:
        if not seed_by_batch:
            print(
                f"[resume] Safe partial Task 5 generation requires math_eval_seed_by_batch=true; "
                f"restarting {policy} from the fixed seed.",
                flush=True,
            )
            records = []
        else:
            keep = (len(records) // batch_size) * batch_size
            if keep != len(records):
                print(
                    f"[resume] Dropping {len(records) - keep} partial {policy} rows "
                    f"to restart the interrupted batch at {keep}",
                    flush=True,
                )
                records = records[:keep]
            print(
                f"[resume] Continuing {dataset} generation for {policy} from "
                f"{len(records)}/{len(rows)} responses",
                flush=True,
            )
        write_jsonl(path, records)

    model = load_frozen_policy(cfg, policy)
    try:
        if not seed_by_batch:
            set_seed(int(cfg["seed"]))

        for start in range(len(records), len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            if seed_by_batch:
                set_seed(int(cfg["seed"]) + start)

            gen = batch_generate(
                model,
                tok,
                [prompt_messages(row) for row in chunk],
                int(cfg.get("math_max_prompt_length", 256)),
                int(cfg["math_max_new_tokens"]),
                **cfg["generation"],
            )

            for local_i, row in enumerate(chunk):
                global_i = start + local_i
                response = gen["responses"][local_i]
                gold = str(row["gold_final"])
                records.append(
                    {
                        "source_index": _source_id(row, global_i),
                        "policy": policy,
                        "question": row.get("question"),
                        "response": response,
                        "response_sha256": _response_hash(response),
                        "response_tokens": int(gen["response_lengths"][local_i]),
                        "gold_final": gold,
                        "predicted_final": extract_designated_final(response),
                        "format_compliant": extract_designated_final(response) is not None,
                        "exact_correct": bool(exact_reward(response, gold)),
                    }
                )

            # Persist each finished batch. A reset loses at most one batch.
            write_jsonl(path, records)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return records


def _load_valid_comparisons(path, generated, ids, policy_names):
    existing = read_jsonl(path) if path.exists() else []
    by_policy_id = {
        name: {str(row["source_index"]): row for row in records}
        for name, records in generated.items()
    }
    valid_pairs = set(combinations(policy_names, 2))
    valid = []
    seen = set()

    for row in existing:
        source = str(row.get("source_index"))
        pair = (row.get("policy_a"), row.get("policy_b"))
        key = (source, *pair)
        if (
            source not in ids
            or pair not in valid_pairs
            or key in seen
            or int(row.get("judge_protocol_version", 0)) != JUDGE_PROTOCOL_VERSION
        ):
            continue

        a_row = by_policy_id[pair[0]].get(source)
        b_row = by_policy_id[pair[1]].get(source)
        if a_row is None or b_row is None:
            continue

        if (
            row.get("a_response_sha256") != _response_hash(a_row["response"])
            or row.get("b_response_sha256") != _response_hash(b_row["response"])
        ):
            continue

        if row.get("preference") not in {"A", "B", "TIE"}:
            continue

        valid.append(row)
        seen.add(key)

    if len(valid) != len(existing):
        write_jsonl(path, valid)
        if existing:
            print(
                f"[resume] Retained {len(valid)}/{len(existing)} pairwise judgments "
                f"that match the current responses and judge protocol.",
                flush=True,
            )
    return valid


def _response_std(records):
    if not records:
        return None
    mean = sum(float(r["response_tokens"]) for r in records) / len(records)
    return (
        sum((float(r["response_tokens"]) - mean) ** 2 for r in records) / len(records)
    ) ** 0.5


def run_evaluation(config_path: str, dataset: str):
    cfg, rows, tok = load_math_evaluation(config_path, dataset)
    if dataset == "transfer" and len(rows) != 100:
        raise ValueError("Fixed transfer evaluation must contain exactly 100 examples")
    if not rows:
        raise ValueError(f"Fixed {dataset} evaluation set is empty")

    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)

    generated = {}
    ids = [_source_id(row, i) for i, row in enumerate(rows)]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Fixed {dataset} evaluation contains duplicate source IDs")

    for name in policy_specs(cfg):
        path = outdir / f"{dataset}_{name}_responses.jsonl"
        generated[name] = _generate_policy_responses(
            cfg,
            rows,
            tok,
            dataset,
            name,
            path,
        )

    policy_names = list(policy_specs(cfg))
    comparisons_path = outdir / f"{dataset}_pairwise.jsonl"
    comparisons = _load_valid_comparisons(
        comparisons_path,
        generated,
        set(ids),
        policy_names,
    )
    keyed = {
        (str(row["source_index"]), row["policy_a"], row["policy_b"])
        for row in comparisons
    }
    expected_keys = {
        (ids[i], a, b)
        for i in range(len(rows))
        for a, b in combinations(policy_names, 2)
    }
    missing_keys = expected_keys - keyed

    judge = None
    if missing_keys:
        judge = PairwiseAIJudge(cfg, outdir / "pairwise_judge_cache.json")

    generated_by_id = {
        name: {str(record["source_index"]): record for record in records}
        for name, records in generated.items()
    }

    try:
        for i, row in enumerate(rows):
            source = ids[i]
            problem = str(row.get("question", prompt_messages(row)[-1]["content"]))
            for a, b in combinations(policy_names, 2):
                key = (source, a, b)
                if key in keyed:
                    continue

                a_record = generated_by_id[a][source]
                b_record = generated_by_id[b][source]
                preference = judge.compare(
                    problem,
                    a_record["response"],
                    b_record["response"],
                )
                record = {
                    "source_index": source,
                    "policy_a": a,
                    "policy_b": b,
                    "preference": preference,
                    "a_correct": bool(a_record["exact_correct"]),
                    "b_correct": bool(b_record["exact_correct"]),
                    "a_response_sha256": _response_hash(a_record["response"]),
                    "b_response_sha256": _response_hash(b_record["response"]),
                    "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
                }
                comparisons.append(record)
                keyed.add(key)
                write_jsonl(comparisons_path, comparisons)
    finally:
        if judge is not None:
            del judge
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metrics = {
        "dataset": dataset,
        "n": len(rows),
        "policies": {},
        "pairwise": {},
        "generation": {
            "max_prompt_length": int(cfg.get("math_max_prompt_length", 256)),
            "max_new_tokens": int(cfg["math_max_new_tokens"]),
            "batch_size": int(cfg.get("math_eval_batch_size", 1)),
            "seed_scheme": (
                "seed + batch_start_index"
                if bool(cfg.get("math_eval_seed_by_batch", False))
                else "single global seed"
            ),
        },
        "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
        "pairwise_score_definition": "win=1, tie=0.5, loss=0",
        "verifier_judge_agreement_definition":
            "Verifier chooses A/B when exactly one final answer is correct, otherwise TIE. "
            "Strict agreement includes judge ties; decisive agreement excludes them and uses only verifier disagreements.",
    }

    for name, records in generated.items():
        mean_tokens = sum(r["response_tokens"] for r in records) / len(records)
        metrics["policies"][name] = {
            "n": len(records),
            "exact_accuracy": sum(r["exact_correct"] for r in records) / len(records),
            "format_compliance": sum(r["format_compliant"] for r in records) / len(records),
            "mean_response_tokens": mean_tokens,
            "response_tokens_std": _response_std(records),
        }

    for a, b in combinations(policy_names, 2):
        subset = [
            r for r in comparisons
            if r["policy_a"] == a and r["policy_b"] == b
        ]
        if len(subset) != len(rows):
            raise RuntimeError(
                f"Expected {len(rows)} {a}_vs_{b} judgments for {dataset}; found {len(subset)}"
            )

        diagnostic = [r for r in subset if r["a_correct"] != r["b_correct"]]
        decisive = [r for r in diagnostic if r["preference"] != "TIE"]

        def verifier_preference(row):
            if row["a_correct"] and not row["b_correct"]:
                return "A"
            if row["b_correct"] and not row["a_correct"]:
                return "B"
            return "TIE"

        metrics["pairwise"][f"{a}_vs_{b}"] = {
            "n": len(subset),
            "a_win_rate": sum(r["preference"] == "A" for r in subset) / len(subset),
            "b_win_rate": sum(r["preference"] == "B" for r in subset) / len(subset),
            "tie_rate": sum(r["preference"] == "TIE" for r in subset) / len(subset),
            "a_pairwise_score": sum(
                1.0 if r["preference"] == "A"
                else 0.5 if r["preference"] == "TIE"
                else 0.0
                for r in subset
            ) / len(subset),
            "b_pairwise_score": sum(
                1.0 if r["preference"] == "B"
                else 0.5 if r["preference"] == "TIE"
                else 0.0
                for r in subset
            ) / len(subset),
            "verifier_judge_strict_three_way_agreement": sum(
                r["preference"] == verifier_preference(r) for r in subset
            ) / len(subset),
            "verifier_judge_strict_agreement_on_verifier_disagreements": (
                sum(r["preference"] == verifier_preference(r) for r in diagnostic)
                / len(diagnostic)
                if diagnostic else None
            ),
            "verifier_judge_decisive_agreement_on_verifier_disagreements": (
                sum(r["preference"] == verifier_preference(r) for r in decisive)
                / len(decisive)
                if decisive else None
            ),
            "verifier_disagreements": len(diagnostic),
            "judge_decisions_on_verifier_disagreements": len(decisive),
            "judge_ties_on_verifier_disagreements": sum(
                r["preference"] == "TIE" for r in diagnostic
            ),
        }

    save_json(outdir / f"{dataset}_metrics.json", metrics)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    args = ap.parse_args()
    print(run_evaluation(args.config, args.dataset))


if __name__ == "__main__":
    main()
