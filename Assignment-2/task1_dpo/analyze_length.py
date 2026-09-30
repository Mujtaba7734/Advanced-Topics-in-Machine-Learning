from __future__ import annotations

import argparse
import gc
from collections import defaultdict
import torch
from common.data import load_yaml, read_jsonl, repo_path, prompt_messages, write_jsonl
from common.logging_utils import save_json, set_seed
from common.metrics import word_limit_compliance, word_count
from common.models import load_policy, load_tokenizer
from common.generation import batch_generate
from task1_dpo.train import run_training
from task1_dpo.evaluate import run_evaluation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    run_training(args.config, "length_balanced", dataset_path=cfg["paths"]["dpo_length_train"],
                 output_path=cfg["length_output"])
    results = {}
    for name, adapter in (("standard", cfg["standard_output"]), ("length_balanced", cfg["length_output"])):
        results[name] = run_evaluation(args.config, adapter, f"{name}_stratified", rows=stratified)
        examples = read_jsonl(repo_path(cfg["results_dir"]) / f"{name}_stratified_examples.jsonl")
        grouped = defaultdict(list)
        for row in examples:
            key = next((str(row[k]) for k in ("length_stratum", "length_bucket", "length_bin",
                        "stratum", "bucket", "category") if k in row), None)
            if key is None:
                raise ValueError("Length-stratified evaluation rows lack a stratum label")
            grouped[key].append(row)
        results[name]["strata"] = {key: {"n": len(group),
            "preference_accuracy": sum(x["preference_correct"] for x in group) / len(group),
            "mean_reward_margin": sum(x["chosen_reward"] - x["rejected_reward"] for x in group) / len(group),
            "mean_response_tokens": sum(x["response_tokens"] for x in group) / len(group)}
            for key, group in grouped.items()}
    prompts = read_jsonl(cfg["paths"]["word_limit_prompts"])
    word_results = {}
    for name, adapter in (("standard", cfg["standard_output"]), ("length_balanced", cfg["length_output"])):
        set_seed(int(cfg["seed"]))
        tok, model = load_tokenizer(cfg["base_model"]), load_policy(cfg, adapter_path=adapter)
        rows = []
        for row in prompts:
            messages = prompt_messages(row)
            gen = batch_generate(model, tok, [messages], int(cfg["max_sequence_length"]),
                                 int(cfg["max_generation_tokens"]), **cfg["generation"])
            response = gen["responses"][0]
            prompt = str(row.get("prompt", row.get("question", messages[-1]["content"])))
            rows.append({"source_index": row.get("source_index", row.get("prompt_id")),
                         "prompt": prompt, "response": response, "word_count": word_count(response),
                         "compliant": word_limit_compliance(prompt, response)})
        write_jsonl(repo_path(cfg["results_dir"]) / f"{name}_word_limits.jsonl", rows)
        values = [r["compliant"] for r in rows if r["compliant"] is not None]
        word_results[name] = {"n": len(values), "compliance_rate": sum(values) / len(values) if values else None}
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save_json(repo_path(cfg["results_dir"]) / "length_study.json", {"stratified": results, "word_limits": word_results})


if __name__ == "__main__":
    main()
