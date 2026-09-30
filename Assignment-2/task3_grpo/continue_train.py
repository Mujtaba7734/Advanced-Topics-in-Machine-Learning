from __future__ import annotations

import argparse
import time
import torch

from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from task3_grpo.grpo import grpo_policy_loss, group_relative_advantages, mask_truncated_sequences


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None,
             loss_type: str = "grpo", run_name: str = "standard",
             log_optimization_details: bool = False):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    policy, tok, optimizer = bundle["policy"], bundle["tokenizer"], bundle["optimizer"]
    prompts = bundle["prompt_rows"]
    k = int(cfg["num_generations"])
    n = int(cfg["updates"])
    log_path = repo_path(cfg["results_dir"]) / f"{run_name}_train.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    details_path = repo_path(cfg["results_dir"]) / f"{run_name}_optimization.jsonl"
    if log_optimization_details:
        details_path.write_text("", encoding="utf-8")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for update in range(n):
        indices = [(update * int(cfg["prompts_per_update"]) + j) % len(prompts)
                   for j in range(int(cfg["prompts_per_update"]))]
        messages = [prompt_messages(prompts[i]) for i in indices for _ in range(k)]
        gen = batch_generate(policy, tok, messages, int(cfg["max_prompt_length"]),
            int(cfg["max_completion_length"]), **cfg["generation"])
        seq, attn, response_ids = gen["sequences"], gen["attention_mask"], gen["response_ids"]
        mask = gen["response_mask"]
        if cfg.get("mask_truncated_completions", False):
            mask = mask_truncated_sequences(mask, gen["truncated"])
        width = gen["prompt_width"]
        with torch.no_grad():
            policy.eval()
            old_logp, _ = response_token_logprobs(policy, seq, attn, width, response_ids)
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, seq, attn, width, response_ids)
            rewards = score_reward_pairs(bundle["reward_model"], bundle["reward_tokenizer"],
                messages, gen["responses"], int(cfg["max_prompt_length"]) + int(cfg["max_completion_length"]))
            group_ids = torch.arange(len(indices), device=rewards.device).repeat_interleave(k)
            advantages = group_relative_advantages(rewards, group_ids)
        policy.train()
        for policy_epoch in range(int(cfg["policy_epochs"])):
            optimizer.zero_grad(set_to_none=True)
            new_logp, _ = response_token_logprobs(policy, seq, attn, width, response_ids)
            loss, diag = grpo_policy_loss(new_logp, old_logp, advantages, mask, ref_logp,
                float(cfg["clip_epsilon"]), float(cfg["kl_beta"]), loss_type,
                int(cfg["max_completion_length"]))
            if log_optimization_details:
                with torch.no_grad():
                    ratio = torch.exp(new_logp - old_logp)
                    clipped = ratio.clamp(1.0 - float(cfg["clip_epsilon"]),
                                          1.0 + float(cfg["clip_epsilon"]))
                    token_surrogate = torch.minimum(ratio * advantages[:, None],
                                                    clipped * advantages[:, None])
                    lengths = mask.sum(-1)
                    denom = (lengths.clamp_min(1.0) if loss_type == "grpo"
                             else torch.full_like(lengths, float(cfg["max_completion_length"])))
                    normalized = (token_surrogate * mask).sum(-1) / denom
                    for j in range(len(messages)):
                        append_jsonl(details_path, {
                            "update": update + 1, "policy_epoch": policy_epoch + 1,
                            "source_index": prompts[indices[j // k]].get("source_index", indices[j // k]),
                            "generation_index": j % k, "loss_type": loss_type,
                            "response_tokens": gen["response_lengths"][j],
                            "optimization_tokens": int(lengths[j]),
                            "truncated": gen["truncated"][j], "advantage": float(advantages[j]),
                            "normalization_weight_per_token": float(1.0 / denom[j]) if lengths[j] > 0 else 0.0,
                            "normalized_clipped_surrogate": float(normalized[j]),
                            "absolute_policy_loss_contribution": float(normalized[j].abs() / len(messages)),
                        })
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"]))
            optimizer.step()
        group_std = rewards.reshape(len(indices), k).std(dim=1, unbiased=False)
        record = {"update": update + 1, "source_indices": [prompts[i].get("source_index", i) for i in indices],
            "reward": float(rewards.mean()), "reward_std": float(group_std.mean()),
            "uninformative_group_fraction": float((group_std <= 1e-6).float().mean()),
            "masked_completion_fraction": sum(gen["truncated"]) / len(gen["truncated"]),
            "response_length": sum(gen["response_lengths"]) / len(gen["response_lengths"]),
            "grad_norm": float(grad_norm), "loss": float(loss), "loss_type": loss_type,
            **{key: float(value) for key, value in diag.items()}}
        append_jsonl(log_path, record)
    policy.save_pretrained(out)
    summary = {"run_name": run_name, "adapter": str(out), "midpoint": cfg["paths"]["grpo_midpoint_policy"],
        "updates": n, "num_generations": k, "loss_type": loss_type,
        "wall_seconds": time.perf_counter() - start,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}
    save_json(repo_path(cfg["results_dir"]) / f"{run_name}_config.json", {**cfg, "loss_type": loss_type})
    save_json(repo_path(cfg["results_dir"]) / f"{run_name}_train.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
