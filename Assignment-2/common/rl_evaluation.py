"""Shared frozen held-out protocol for PPO and GRPO policies."""
from __future__ import annotations

import torch
from tqdm.auto import tqdm

from common.data import prompt_messages, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.metrics import masked_mean, sample_entropy
from common.models import reference_mode


def evaluate_rl(bundle, name, task):
    cfg, tok, policy = bundle["cfg"], bundle["tokenizer"], bundle["policy"]
    rm, rm_tok = bundle["reward"]
    rows = bundle["rows"]
    set_seed(int(cfg["seed"]))
    max_response = int(cfg.get("eval_max_response_length", cfg.get("max_completion_length")))
    examples = []
    for i, row in enumerate(tqdm(rows, total=len(rows), desc=f"{task.upper()} evaluation: {name}", unit="example", dynamic_ncols=True)):
        messages = prompt_messages(row)
        gen = batch_generate(policy, tok, [messages], int(cfg["max_prompt_length"]),
            max_response, **cfg["generation"])
        seq, attn, response_ids = gen["sequences"], gen["attention_mask"], gen["response_ids"]
        mask = gen["response_mask"]
        with torch.no_grad():
            logp, _ = response_token_logprobs(policy, seq, attn, gen["prompt_width"], response_ids)
            with reference_mode(policy):
                ref, _ = response_token_logprobs(policy, seq, attn, gen["prompt_width"], response_ids)
            reward = score_reward_pairs(rm, rm_tok, [messages], gen["responses"],
                int(cfg.get("reward_max_length", int(cfg["max_prompt_length"]) + max_response)))
        examples.append({"source_index": row.get("source_index", i), "response": gen["responses"][0],
            "reward": float(reward[0]), "sampled_kl": float(masked_mean(logp - ref, mask)),
            "entropy": float(sample_entropy(logp, mask)), "response_tokens": gen["response_lengths"][0],
            "terminated_with_eos": gen["terminated_with_eos"][0], "truncated": gen["truncated"][0]})
    def mean(key):
        return sum(float(r[key]) for r in examples) / len(examples) if examples else None
    metrics = {"n": len(examples), "mean_reward": mean("reward"), "mean_sampled_kl": mean("sampled_kl"),
        "mean_entropy": mean("entropy"), "mean_response_tokens": mean("response_tokens"),
        "eos_rate": mean("terminated_with_eos"), "truncation_rate": mean("truncated")}
    out = repo_path(cfg["results_dir"])
    save_json(out / f"{name}_eval.json", metrics)
    write_jsonl(out / f"{name}_examples.jsonl", examples)
    return metrics
