from __future__ import annotations

import argparse
import gc
from itertools import combinations
import torch

from common.data import load_yaml, read_jsonl, repo_path, prompt_messages, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed
from common.models import load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
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


def run_evaluation(config_path: str, dataset: str):
    cfg, rows, tok = load_math_evaluation(config_path, dataset)
    if dataset == "transfer" and len(rows) != 100:
        raise ValueError("Fixed transfer evaluation must contain exactly 100 examples")
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    generated = {}
    ids = [str(r.get("source_index", r.get("prompt_id", i))) for i, r in enumerate(rows)]
    for name in policy_specs(cfg):
        path = outdir / f"{dataset}_{name}_responses.jsonl"
        records = None
        if path.exists():
            records = read_jsonl(path)
            if [str(x["source_index"]) for x in records] != ids[:len(records)]:
                raise ValueError(f"Saved {name} responses do not match fixed {dataset} IDs")
            if len(records) != len(rows):
                records = None  # Recreate an interrupted file from the fixed seed.
        if records is None:
            set_seed(int(cfg["seed"]))
            model = load_frozen_policy(cfg, name)
            records = []
            for i, row in enumerate(rows):
                gen = batch_generate(model, tok, [prompt_messages(row)], 256,
                    int(cfg["math_max_new_tokens"]), **cfg["generation"])
                response = gen["responses"][0]
                records.append({"source_index": ids[i], "policy": name, "question": row.get("question"),
                    "response": response, "response_tokens": gen["response_lengths"][0],
                    "gold_final": str(row["gold_final"]),
                    "predicted_final": extract_designated_final(response),
                    "format_compliant": extract_designated_final(response) is not None,
                    "exact_correct": bool(exact_reward(response, str(row["gold_final"])))})
                write_jsonl(path, records)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        generated[name] = records
    judge = PairwiseAIJudge(cfg, outdir / "pairwise_judge_cache.json")
    comparisons_path = outdir / f"{dataset}_pairwise.jsonl"
    comparisons = read_jsonl(comparisons_path) if comparisons_path.exists() else []
    keyed = {(str(x["source_index"]), x["policy_a"], x["policy_b"]): x for x in comparisons}
    for i, row in enumerate(rows):
        for a, b in combinations(policy_specs(cfg), 2):
            key = (ids[i], a, b)
            if key in keyed:
                continue
            preference = judge.compare(str(row.get("question", prompt_messages(row)[-1]["content"])),
                                       generated[a][i]["response"], generated[b][i]["response"])
            record = {"source_index": ids[i], "policy_a": a, "policy_b": b, "preference": preference,
                "a_correct": generated[a][i]["exact_correct"], "b_correct": generated[b][i]["exact_correct"]}
            comparisons.append(record)
            write_jsonl(comparisons_path, comparisons)
    metrics = {"dataset": dataset, "n": len(rows), "policies": {}, "pairwise": {},
               "pairwise_score_definition": "win=1, tie=0.5, loss=0",
               "verifier_judge_agreement_definition":
                   "Verifier chooses A/B when exactly one final answer is correct, otherwise TIE. "
                   "Strict agreement includes judge ties; decisive agreement excludes them and uses only verifier disagreements."}
    for name, records in generated.items():
        metrics["policies"][name] = {"n": len(records),
            "exact_accuracy": sum(r["exact_correct"] for r in records) / len(records),
            "format_compliance": sum(r["format_compliant"] for r in records) / len(records),
            "mean_response_tokens": sum(r["response_tokens"] for r in records) / len(records)}
    for a, b in combinations(policy_specs(cfg), 2):
        subset = [r for r in comparisons if r["policy_a"] == a and r["policy_b"] == b]
        diagnostic = [r for r in subset if r["a_correct"] != r["b_correct"]]
        decisive = [r for r in diagnostic if r["preference"] != "TIE"]
        def verifier_preference(row):
            return "A" if row["a_correct"] and not row["b_correct"] else (
                "B" if row["b_correct"] and not row["a_correct"] else "TIE")
        metrics["pairwise"][f"{a}_vs_{b}"] = {"n": len(subset),
            "a_win_rate": sum(r["preference"] == "A" for r in subset) / len(subset),
            "b_win_rate": sum(r["preference"] == "B" for r in subset) / len(subset),
            "tie_rate": sum(r["preference"] == "TIE" for r in subset) / len(subset),
            "a_pairwise_score": sum(1.0 if r["preference"] == "A" else
                                    0.5 if r["preference"] == "TIE" else 0.0 for r in subset) / len(subset),
            "b_pairwise_score": sum(1.0 if r["preference"] == "B" else
                                    0.5 if r["preference"] == "TIE" else 0.0 for r in subset) / len(subset),
            "verifier_judge_strict_three_way_agreement":
                sum(r["preference"] == verifier_preference(r) for r in subset) / len(subset),
            "verifier_judge_strict_agreement_on_verifier_disagreements":
                sum(r["preference"] == verifier_preference(r) for r in diagnostic) / len(diagnostic)
                if diagnostic else None,
            "verifier_judge_decisive_agreement_on_verifier_disagreements":
                sum(r["preference"] == verifier_preference(r) for r in decisive) / len(decisive)
                if decisive else None,
            "verifier_disagreements": len(diagnostic),
            "judge_decisions_on_verifier_disagreements": len(decisive),
            "judge_ties_on_verifier_disagreements": sum(r["preference"] == "TIE" for r in diagnostic)}
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
