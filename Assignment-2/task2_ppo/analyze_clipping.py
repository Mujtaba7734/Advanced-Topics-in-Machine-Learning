from __future__ import annotations

import argparse
import statistics
from pathlib import Path
import torch

from common.data import load_yaml, repo_path, read_jsonl, prompt_messages
from common.generation import response_token_logprobs, score_reward_pairs
from common.logging_utils import load_json, save_json
from common.models import load_policy, load_reward_model, load_tokenizer, load_value_model, token_values
from common.rl_evaluation import evaluate_rl
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import load_evaluation_bundle
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards



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


def completed_training(cfg, name: str, adapter: Path, expected_updates: int, expected_eps: float):
    value_adapter = adapter.parent / f"{adapter.name}_value"
    summary_path = repo_path(cfg["results_dir"]) / f"{name}_train.json"
    log_path = repo_path(cfg["results_dir"]) / f"{name}_train.jsonl"
    if not (adapter_is_complete(adapter) and adapter_is_complete(value_adapter)
            and summary_path.exists() and jsonl_count(log_path) == expected_updates):
        return None
    summary = load_json(summary_path)
    if int(summary.get("updates", -1)) != expected_updates:
        return None
    if abs(float(summary.get("clip_epsilon", -999.0)) - float(expected_eps)) > 1e-12:
        return None
    return summary


def completed_evaluation(cfg, name: str, expected_n: int):
    metrics_path = repo_path(cfg["results_dir"]) / f"{name}_eval.json"
    examples_path = repo_path(cfg["results_dir"]) / f"{name}_examples.jsonl"
    if not metrics_path.exists() or jsonl_count(examples_path) != expected_n:
        return None
    metrics = load_json(metrics_path)
    if int(metrics.get("n", -1)) != expected_n:
        return None
    return metrics

def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def analyze_cached_batch(cfg, rows):
    tok = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"])
    value = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode="frozen")
    rm, rm_tok = load_reward_model(cfg)
    prompts = {str(r.get("source_index", i)): r for i, r in enumerate(read_jsonl(cfg["paths"]["rl_prompt_train"]))}
    per_eps = {str(e): [] for e in cfg["clip_values"]}
    for row in rows:
        source = str(row["source_index"])
        if source not in prompts:
            raise ValueError(f"Cached source_index {source} absent from fixed prompt pool")
        messages = prompt_messages(prompts[source])
        old = torch.as_tensor(row["old_logprobs"], dtype=torch.float32).flatten()
        ref = torch.as_tensor(row["ref_logprobs"], dtype=torch.float32).flatten()
        if old.shape != ref.shape:
            raise ValueError("Cached old/reference log-probability lengths disagree")
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(rendered, truncation=True, max_length=int(cfg["max_prompt_length"]))["input_ids"]
        response_ids = row.get("response_ids", row.get("response_token_ids"))
        if response_ids is None:
            response_ids = tok(str(row["response"]), add_special_tokens=False)["input_ids"]
            if len(response_ids) + 1 == len(old):
                response_ids.append(tok.eos_token_id)
        response_ids = [int(x) for x in response_ids]
        if len(response_ids) != len(old):
            raise ValueError(f"Cannot align cached rollout tokens for source_index {source}: {len(response_ids)} vs {len(old)}")
        device = next(policy.parameters()).device
        seq = torch.tensor([prompt_ids + response_ids], device=device)
        attn = torch.ones_like(seq)
        response_tensor = torch.tensor([response_ids], device=device)
        mask = torch.as_tensor(row.get("response_mask", [1] * len(old)), device=device, dtype=torch.float32).reshape(1, -1)
        old, ref = old.to(device)[None, :], ref.to(device)[None, :]
        with torch.no_grad():
            new, _ = response_token_logprobs(policy, seq, attn, len(prompt_ids), response_tensor)
            cached_values = row.get("values", row.get("old_values"))
            if cached_values is not None:
                values = torch.as_tensor(cached_values, device=device, dtype=torch.float32).reshape(1, -1)
            else:
                values = token_values(value, seq, attn)[:, len(prompt_ids) - 1:-1].float()
            if values.shape[-1] == old.shape[-1] + 1:
                values = values[:, :-1]
            if values.shape != old.shape:
                raise ValueError(f"Cached value length mismatch for source_index {source}")
            if any(key in row for key in ("task_reward", "reward", "score")):
                task_reward = torch.tensor([float(next(row[key] for key in ("task_reward", "reward", "score") if key in row))], device=device)
            else:
                task_reward = score_reward_pairs(rm, rm_tok, [messages], [row["response"]], int(cfg["reward_max_length"]))
            if not row.get("terminated_with_eos", tok.eos_token_id in response_ids):
                task_reward -= float(cfg["missing_eos_penalty"])
            rewards = shaped_rewards(task_reward, old, ref, mask, cfg["kl_beta"])
            advantage, returns = compute_gae(rewards, values, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
            advantage = normalize_advantages(advantage, mask)
            for eps in cfg["clip_values"]:
                loss, ratio, fraction = ppo_policy_loss(new, old, advantage, mask, float(eps))
                per_eps[str(eps)].append({"source_index": source, "tokens": int(mask.sum()),
                    "surrogate": -float(loss), "affected_token_fraction": float(fraction),
                    "mean_ratio": float((ratio * mask).sum() / mask.sum().clamp_min(1)),
                    "mean_return": float((returns * mask).sum() / mask.sum().clamp_min(1))})
    return {key: {"n_rollouts": len(items), "tokens": sum(x["tokens"] for x in items),
        "mean_surrogate": sum(x["surrogate"] * x["tokens"] for x in items) / max(1, sum(x["tokens"] for x in items)),
        "affected_token_fraction": sum(x["affected_token_fraction"] * x["tokens"] for x in items) / max(1, sum(x["tokens"] for x in items)),
        "rollouts": items} for key, items in per_eps.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse completed PPO clipping forks/evaluations.")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    study_path = repo_path(cfg["results_dir"]) / "clipping_study.json"

    cached = None
    if args.resume and study_path.exists():
        previous = load_json(study_path)
        cached = previous.get("cached")
        if cached is not None:
            print("[resume] Reusing completed cached-rollout clipping analysis", flush=True)
    if cached is None:
        rows = load_cached_rollouts(cfg["cached_rollouts"])
        cached = analyze_cached_batch(cfg, rows)

    expected_eval_n = len(read_jsonl(cfg["paths"]["rl_prompt_eval"]))
    forks = []
    for eps in cfg["clip_values"]:
        name = f"clip_{float(eps):.2f}"
        adapter = repo_path(cfg["output"]).parent / name
        train = completed_training(
            cfg, name, adapter, int(cfg["fork_updates"]), float(eps)
        ) if args.resume else None
        if train is not None:
            print(f"[resume] Reusing completed training: {name}", flush=True)
        else:
            train = run_ppo(
                args.config, str(adapter), int(cfg["fork_updates"]),
                clip_epsilon=float(eps), run_name=name
            )

        evaluation = completed_evaluation(cfg, name, expected_eval_n) if args.resume else None
        if evaluation is not None:
            print(f"[resume] Reusing completed evaluation: {name}", flush=True)
        else:
            evaluation = evaluate_rl(
                load_evaluation_bundle(args.config, str(adapter)), name, "ppo"
            )

        trajectory = read_jsonl(repo_path(cfg["results_dir"]) / f"{name}_train.jsonl")
        grad_norms = [float(row["policy_grad_norm"]) for row in trajectory]
        policy_losses = [float(row["policy_loss"]) for row in trajectory]
        stability = {
            "n_updates": len(trajectory),
            "policy_grad_norm_std": statistics.pstdev(grad_norms) if grad_norms else None,
            "policy_loss_std": statistics.pstdev(policy_losses) if policy_losses else None,
            "max_policy_grad_norm": max(grad_norms) if grad_norms else None,
        }
        forks.append({
            "clip_epsilon": float(eps),
            "train": train,
            "evaluation": evaluation,
            "stability": stability,
        })
        save_json(study_path, {"cached": cached, "forks": forks})


if __name__ == "__main__":
    main()
