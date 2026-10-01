from __future__ import annotations

import argparse
import time
import torch

from torch.optim import AdamW
from tqdm.auto import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import masked_mean, sample_entropy
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    trainable_parameters,
    value_parameter_groups,
    reference_mode,
    token_values,
)
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    policy, value = bundle["policy"], bundle["value_model"]
    po, vo = bundle["policy_optimizer"], bundle["value_optimizer"]
    tok = bundle["tokenizer"]
    prompts = bundle["prompt_rows"]
    n = int(cfg["updates"])
    log_path = repo_path(cfg["results_dir"]) / f"{run_name}_train.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    progress = tqdm(range(n), total=n, desc=f"PPO {run_name}", unit="update", dynamic_ncols=True)
    for update in progress:
        indices = [(update * int(cfg["prompts_per_update"]) + j) % len(prompts)
                   for j in range(int(cfg["prompts_per_update"]))]
        messages = [prompt_messages(prompts[i]) for i in indices]
        gen = batch_generate(policy, tok, messages, int(cfg["max_prompt_length"]),
            int(cfg["max_response_length"]), **cfg["generation"])
        seq, attn, response_ids = gen["sequences"], gen["attention_mask"], gen["response_ids"]
        mask = gen["response_mask"]
        width = gen["prompt_width"]
        with torch.no_grad():
            policy.eval()
            old_logp, _ = response_token_logprobs(policy, seq, attn, width, response_ids)
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, seq, attn, width, response_ids)
            value.eval()
            old_values = token_values(value, seq, attn)[:, width - 1:-1].float()
            task_reward = score_reward_pairs(bundle["reward_model"], bundle["reward_tokenizer"],
                messages, gen["responses"], int(cfg["reward_max_length"]))
            task_reward -= float(cfg["missing_eos_penalty"]) * torch.tensor(
                [not x for x in gen["terminated_with_eos"]], device=task_reward.device, dtype=task_reward.dtype)
            rewards = shaped_rewards(task_reward, old_logp, ref_logp, mask, cfg["kl_beta"])
            advantages, returns = compute_gae(rewards, old_values, mask,
                float(cfg["gamma"]), float(cfg["gae_lambda"]))
            advantages = normalize_advantages(advantages, mask).detach()
            returns = returns.detach()
        # PPO ratios must compare the old and updated policy under the same
        # deterministic network. eval() disables LoRA dropout but does NOT
        # disable autograd, so the adapters remain fully trainable.
        policy.eval()
        value.eval()
        for _ in range(int(cfg["ppo_epochs"])):
            po.zero_grad(set_to_none=True)
            vo.zero_grad(set_to_none=True)
            new_logp, _ = response_token_logprobs(policy, seq, attn, width, response_ids)
            ploss, ratio, clip_frac = ppo_policy_loss(new_logp, old_logp, advantages, mask, float(cfg["clip_epsilon"]))
            if not torch.isfinite(ploss):
                raise FloatingPointError(
                    f"Non-finite PPO policy loss at update {update + 1}; "
                    f"ratio range=({float(ratio.min()):.6g}, {float(ratio.max()):.6g})"
                )
            ploss.backward()
            policy_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"]))
            if not torch.isfinite(policy_norm):
                raise FloatingPointError(f"Non-finite PPO policy gradient norm at update {update + 1}")
            po.step()

            predicted = token_values(value, seq, attn)[:, width - 1:-1].float()
            vloss = value_mse_loss(predicted, returns, mask)
            if not torch.isfinite(vloss):
                raise FloatingPointError(f"Non-finite PPO value loss at update {update + 1}")
            (float(cfg["value_coef"]) * vloss).backward()
            value_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(value), float(cfg["max_grad_norm"]))
            if not torch.isfinite(value_norm):
                raise FloatingPointError(f"Non-finite PPO value gradient norm at update {update + 1}")
            vo.step()
        record = {"update": update + 1, "source_indices": [prompts[i].get("source_index", i) for i in indices],
            "reward": float(task_reward.mean()), "sampled_kl": float(masked_mean(old_logp - ref_logp, mask)),
            "policy_loss": float(ploss.detach()), "value_loss": float(vloss.detach()),
            "entropy": float(sample_entropy(new_logp.detach(), mask)),
            "policy_grad_norm": float(policy_norm.detach()), "value_grad_norm": float(value_norm.detach()),
            "clip_fraction": float(clip_frac), "response_length": float(mask.sum(-1).mean()),
            "ratio_mean": float(((ratio * mask).sum() / mask.sum().clamp_min(1)).detach()),
            "ratio_min": float(ratio[mask.bool()].min().detach()),
            "ratio_max": float(ratio[mask.bool()].max().detach()),
            "eos_rate": sum(gen["terminated_with_eos"]) / len(indices)}
        append_jsonl(log_path, record)
        progress.set_postfix(
            reward=f"{record['reward']:.3f}",
            kl=f"{record['sampled_kl']:.4f}",
            ploss=f"{record['policy_loss']:.4f}",
            vloss=f"{record['value_loss']:.4f}",
        )
    policy.save_pretrained(out)
    value_dir = out.parent / f"{out.name}_value"
    value.save_pretrained(value_dir)
    summary = {"run_name": run_name, "updates": n, "adapter": str(out),
        "value_adapter": str(value_dir), "midpoint_policy": cfg["paths"]["ppo_midpoint_policy"],
        "midpoint_value": cfg["paths"]["ppo_midpoint_value"],
        "clip_epsilon": cfg["clip_epsilon"], "kl_beta": cfg["kl_beta"],
        "wall_seconds": time.perf_counter() - start,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}
    save_json(repo_path(cfg["results_dir"]) / f"{run_name}_config.json", cfg)
    save_json(repo_path(cfg["results_dir"]) / f"{run_name}_train.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
