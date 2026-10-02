from __future__ import annotations

import argparse
import time

import torch
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task3_grpo.grpo import (
    grpo_policy_loss,
    group_relative_advantages,
    mask_truncated_sequences,
)


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


def _deterministic_train_mode(policy) -> int:
    """Keep autograd/checkpointing enabled while disabling stochastic dropout."""
    policy.train()
    disabled = 0
    for module in policy.modules():
        if isinstance(module, nn.Dropout):
            module.eval()
            disabled += 1
    return disabled


def _require_finite(
    name: str,
    tensor: torch.Tensor,
    update: int,
    policy_epoch: int | None = None,
) -> None:
    if torch.isfinite(tensor).all():
        return
    where = f"update {update}"
    if policy_epoch is not None:
        where += f", policy epoch {policy_epoch}"
    raise FloatingPointError(f"Non-finite {name} at {where}")


def run_grpo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    loss_type: str = "grpo",
    run_name: str = "standard",
    log_optimization_details: bool = False,
):
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
    prompts_per_update = int(cfg["prompts_per_update"])
    policy_epochs = int(cfg["policy_epochs"])
    max_completion_length = int(cfg["max_completion_length"])

    if k < 2:
        raise ValueError("GRPO requires at least two generations per prompt")
    if n <= 0 or prompts_per_update <= 0 or policy_epochs <= 0:
        raise ValueError("GRPO updates/prompts_per_update/policy_epochs must be positive")

    log_path = repo_path(cfg["results_dir"]) / f"{run_name}_train.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    details_path = repo_path(cfg["results_dir"]) / f"{run_name}_optimization.jsonl"
    if log_optimization_details:
        details_path.write_text("", encoding="utf-8")

    trainable = trainable_parameters(policy)
    if not trainable:
        raise RuntimeError("GRPO policy has no trainable parameters")
    for parameter in trainable:
        _require_finite(
            "initial trainable policy parameter",
            parameter.detach(),
            update=0,
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    disabled_dropout_modules = _deterministic_train_mode(policy)
    start = time.perf_counter()
    progress = tqdm(
        range(n),
        total=n,
        desc=f"GRPO {run_name}",
        unit="update",
        dynamic_ncols=True,
    )

    for update_zero in progress:
        update = update_zero + 1
        indices = [
            (update_zero * prompts_per_update + j) % len(prompts)
            for j in range(prompts_per_update)
        ]
        messages = [
            prompt_messages(prompts[i])
            for i in indices
            for _ in range(k)
        ]

        gen = batch_generate(
            policy,
            tok,
            messages,
            int(cfg["max_prompt_length"]),
            max_completion_length,
            **cfg["generation"],
        )
        seq = gen["sequences"]
        attn = gen["attention_mask"]
        response_ids = gen["response_ids"]
        mask = gen["response_mask"]
        if cfg.get("mask_truncated_completions", False):
            mask = mask_truncated_sequences(mask, gen["truncated"])
        width = gen["prompt_width"]

        with torch.no_grad():
            # batch_generate restores train mode. Disable dropout before old-policy
            # likelihoods so the denominator of the importance ratio is deterministic.
            _deterministic_train_mode(policy)
            old_logp, _ = response_token_logprobs(
                policy,
                seq,
                attn,
                width,
                response_ids,
            )
            _require_finite("old policy log-probabilities", old_logp, update)

            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    policy,
                    seq,
                    attn,
                    width,
                    response_ids,
                )
            _require_finite("reference log-probabilities", ref_logp, update)

            rewards = score_reward_pairs(
                bundle["reward_model"],
                bundle["reward_tokenizer"],
                messages,
                gen["responses"],
                int(cfg["max_prompt_length"]) + max_completion_length,
            )
            _require_finite("reward", rewards, update)

            group_ids = torch.arange(
                len(indices),
                device=rewards.device,
            ).repeat_interleave(k)
            advantages = group_relative_advantages(rewards, group_ids)
            _require_finite("group-relative advantages", advantages, update)

        # reference_mode restores model.train(), including dropout submodules.
        # Re-disable dropout while keeping the parent model in training mode so
        # gradient checkpointing remains available for K=4 training.
        _deterministic_train_mode(policy)

        grad_norm = None
        loss = None
        diag = None
        ratio = None

        for policy_epoch_zero in range(policy_epochs):
            policy_epoch = policy_epoch_zero + 1
            optimizer.zero_grad(set_to_none=True)
            _deterministic_train_mode(policy)

            new_logp, _ = response_token_logprobs(
                policy,
                seq,
                attn,
                width,
                response_ids,
            )
            _require_finite(
                "new policy log-probabilities",
                new_logp,
                update,
                policy_epoch,
            )

            loss, diag = grpo_policy_loss(
                new_logp,
                old_logp,
                advantages,
                mask,
                ref_logp,
                float(cfg["clip_epsilon"]),
                float(cfg["kl_beta"]),
                loss_type,
                max_completion_length,
            )
            _require_finite("GRPO loss", loss, update, policy_epoch)

            ratio = torch.exp(new_logp.detach() - old_logp)
            _require_finite("policy ratio", ratio, update, policy_epoch)

            if log_optimization_details:
                with torch.no_grad():
                    clipped = ratio.clamp(
                        1.0 - float(cfg["clip_epsilon"]),
                        1.0 + float(cfg["clip_epsilon"]),
                    )
                    token_surrogate = torch.minimum(
                        ratio * advantages[:, None],
                        clipped * advantages[:, None],
                    )
                    lengths = mask.sum(-1)
                    if loss_type == "grpo":
                        denom = lengths.clamp_min(1.0)
                    else:
                        denom = torch.full_like(
                            lengths,
                            float(max_completion_length),
                        )
                    normalized = (token_surrogate * mask).sum(-1) / denom

                    for j in range(len(messages)):
                        append_jsonl(
                            details_path,
                            {
                                "update": update,
                                "policy_epoch": policy_epoch,
                                "source_index": prompts[indices[j // k]].get(
                                    "source_index",
                                    indices[j // k],
                                ),
                                "generation_index": j % k,
                                "loss_type": loss_type,
                                "response_tokens": int(gen["response_lengths"][j]),
                                "optimization_tokens": int(lengths[j].item()),
                                "truncated": bool(gen["truncated"][j]),
                                "advantage": float(advantages[j].detach()),
                                "normalization_weight_per_token": (
                                    float(1.0 / denom[j].item())
                                    if lengths[j].item() > 0
                                    else 0.0
                                ),
                                "normalized_clipped_surrogate": float(
                                    normalized[j].detach()
                                ),
                                "absolute_policy_loss_contribution": float(
                                    normalized[j].detach().abs() / len(messages)
                                ),
                            },
                        )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                float(cfg["max_grad_norm"]),
            )
            _require_finite(
                "policy gradient norm",
                torch.as_tensor(grad_norm),
                update,
                policy_epoch,
            )
            optimizer.step()

            for parameter in trainable:
                _require_finite(
                    "trainable policy parameter after optimizer step",
                    parameter.detach(),
                    update,
                    policy_epoch,
                )

        if loss is None or diag is None or grad_norm is None or ratio is None:
            raise RuntimeError("GRPO optimization loop executed zero policy epochs")

        group_std = rewards.reshape(len(indices), k).std(
            dim=1,
            unbiased=False,
        )
        valid_ratio = ratio[mask.bool()]
        ratio_min = float(valid_ratio.min()) if valid_ratio.numel() else None
        ratio_max = float(valid_ratio.max()) if valid_ratio.numel() else None

        record = {
            "update": update,
            "source_indices": [
                prompts[i].get("source_index", i)
                for i in indices
            ],
            "reward": float(rewards.mean()),
            "reward_std": float(group_std.mean()),
            "uninformative_group_fraction": float(
                (group_std <= 1e-6).float().mean()
            ),
            "masked_completion_fraction": float(
                sum(bool(x) for x in gen["truncated"]) / len(gen["truncated"])
            ),
            "response_length": float(
                sum(gen["response_lengths"]) / len(gen["response_lengths"])
            ),
            "generated_tokens": int(sum(gen["response_lengths"])),
            "optimization_tokens": int(mask.sum().item()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach()),
            "loss": float(loss.detach()),
            "loss_type": loss_type,
            "ratio_min": ratio_min,
            "ratio_max": ratio_max,
            **{key: float(value) for key, value in diag.items()},
        }
        append_jsonl(log_path, record)

        progress.set_postfix(
            reward=f"{record['reward']:.3f}",
            kl=f"{record['sampled_kl']:.4f}",
            loss=f"{record['loss']:.4f}",
            masked=f"{record['masked_completion_fraction']:.2f}",
        )

    policy.save_pretrained(out)

    max_generated_token_budget = (
        n
        * prompts_per_update
        * k
        * max_completion_length
    )
    summary = {
        "run_name": run_name,
        "adapter": str(out),
        "midpoint": cfg["paths"]["grpo_midpoint_policy"],
        "updates": n,
        "prompts_per_update": prompts_per_update,
        "num_generations": k,
        "policy_epochs": policy_epochs,
        "loss_type": loss_type,
        "clip_epsilon": float(cfg["clip_epsilon"]),
        "kl_beta": float(cfg["kl_beta"]),
        "max_completion_length": max_completion_length,
        "mask_truncated_completions": bool(
            cfg.get("mask_truncated_completions", False)
        ),
        "max_generated_token_budget": max_generated_token_budget,
        "dropout_modules_disabled_during_ratio_updates": (
            disabled_dropout_modules
        ),
        "wall_seconds": time.perf_counter() - start,
        "peak_vram_bytes": (
            torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else None
        ),
    }
    save_json(
        repo_path(cfg["results_dir"]) / f"{run_name}_config.json",
        {**cfg, "loss_type": loss_type},
    )
    save_json(
        repo_path(cfg["results_dir"]) / f"{run_name}_train.json",
        summary,
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument(
        "--loss-type",
        choices=["grpo", "dr_grpo"],
        default="grpo",
    )
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--log-optimization-details", action="store_true")
    args = ap.parse_args()
    run_grpo(
        args.config,
        args.output,
        args.updates,
        args.loss_type,
        args.run_name,
        log_optimization_details=args.log_optimization_details,
    )


if __name__ == "__main__":
    main()
