from __future__ import annotations

import argparse
import gc
from collections import defaultdict
from pathlib import Path

import torch

from common.data import load_yaml, read_jsonl, repo_path, prompt_messages, write_jsonl
from common.logging_utils import load_json, save_json, set_seed
from common.metrics import word_limit_compliance, word_count
from common.models import load_policy, load_tokenizer
from common.generation import batch_generate
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
                    help="Reuse completed length-study training/evaluations and continue from the first incomplete stage.")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])

    length_adapter = repo_path(cfg["length_output"])
    length_train_path = repo_path(cfg["results_dir"]) / "length_balanced_train.json"
    if args.resume and adapter_is_complete(length_adapter) and length_train_path.exists():
        print("[resume] Reusing completed length-balanced training", flush=True)
    else:
        run_training(
            args.config,
            "length_balanced",
            dataset_path=cfg["paths"]["dpo_length_train"],
            output_path=cfg["length_output"],
        )

    results = {}
    for name, adapter in (("standard", cfg["standard_output"]), ("length_balanced", cfg["length_output"])):
        eval_name = f"{name}_stratified"
        evaluation = completed_evaluation(cfg, eval_name, len(stratified)) if args.resume else None
        if evaluation is not None:
            print(f"[resume] Reusing completed evaluation: {eval_name}", flush=True)
            results[name] = evaluation
        else:
            results[name] = run_evaluation(args.config, adapter, eval_name, rows=stratified)

        examples = read_jsonl(repo_path(cfg["results_dir"]) / f"{eval_name}_examples.jsonl")
        grouped = defaultdict(list)
        for row in examples:
            key = next((str(row[k]) for k in (
                "length_stratum", "length_bucket", "length_bin", "stratum", "bucket", "category"
            ) if k in row), None)
            if key is None:
                raise ValueError("Length-stratified evaluation rows lack a stratum label")
            grouped[key].append(row)

        results[name]["strata"] = {
            key: {
                "n": len(group),
                "preference_accuracy": sum(x["preference_correct"] for x in group) / len(group),
                "mean_reward_margin": sum(x["chosen_reward"] - x["rejected_reward"] for x in group) / len(group),
                "mean_response_tokens": sum(x["response_tokens"] for x in group) / len(group),
            }
            for key, group in grouped.items()
        }

    prompts = read_jsonl(cfg["paths"]["word_limit_prompts"])
    word_results = {}
    for name, adapter in (("standard", cfg["standard_output"]), ("length_balanced", cfg["length_output"])):
        word_path = repo_path(cfg["results_dir"]) / f"{name}_word_limits.jsonl"
        if args.resume and jsonl_count(word_path) == len(prompts):
            print(f"[resume] Reusing completed word-limit evaluation: {name}", flush=True)
            rows = read_jsonl(word_path)
        else:
            set_seed(int(cfg["seed"]))
            tok, model = load_tokenizer(cfg["base_model"]), load_policy(cfg, adapter_path=adapter)
            rows = []
            for row in prompts:
                messages = prompt_messages(row)
                gen = batch_generate(
                    model,
                    tok,
                    [messages],
                    int(cfg["max_sequence_length"]),
                    int(cfg["max_generation_tokens"]),
                    **cfg["generation"],
                )
                response = gen["responses"][0]
                prompt = str(row.get("prompt", row.get("question", messages[-1]["content"])))
                rows.append({
                    "source_index": row.get("source_index", row.get("prompt_id")),
                    "prompt": prompt,
                    "response": response,
                    "word_count": word_count(response),
                    "compliant": word_limit_compliance(prompt, response),
                })
                # Persist partial progress after every prompt.
                write_jsonl(word_path, rows)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        values = [r["compliant"] for r in rows if r["compliant"] is not None]
        word_results[name] = {
            "n": len(values),
            "compliance_rate": sum(values) / len(values) if values else None,
        }

    save_json(
        repo_path(cfg["results_dir"]) / "length_study.json",
        {"stratified": results, "word_limits": word_results},
    )


if __name__ == "__main__":
    main()
