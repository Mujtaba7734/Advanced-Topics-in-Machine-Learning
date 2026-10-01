from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from common.data import (encode_prompt_response, load_yaml, pad_batch, preference_responses,
                         prompt_messages_from_preference, read_jsonl, repo_path, write_jsonl)
from common.generation import (batch_generate, response_sequence_logprobs,
                               response_token_logprobs, score_reward_pairs)
from common.logging_utils import save_json, set_seed
from common.metrics import word_count
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def _source_id(row, fallback):
    return str(row.get("source_index", fallback))


def _summarize(examples, cfg, generate):
    def mean(key):
        return sum(float(x[key]) for x in examples) / len(examples) if examples else None

    length_mean = mean("response_tokens") if generate else None
    token_total = sum(x["response_tokens"] for x in examples) if generate else 0
    return {
        "n": len(examples),
        "preference_accuracy": mean("preference_correct"),
        "reference_adjusted_accuracy": mean("preference_correct"),
        "policy_only_preference_accuracy": mean("policy_only_preference_correct"),
        "heldout_dpo_loss": mean("heldout_dpo_loss"),
        "beta": float(cfg["beta"]),
        "mean_reference_log_ratio_chosen":
            (sum(x["policy_chosen_logp"] - x["reference_chosen_logp"] for x in examples) / len(examples))
            if examples else None,
        "mean_reward_margin":
            (sum(x["chosen_reward"] - x["rejected_reward"] for x in examples) / len(examples))
            if examples else None,
        "mean_response_reward": mean("response_reward") if generate else None,
        "sampled_response_kl":
            (sum(x["sampled_response_kl"] * x["response_tokens"] for x in examples) / token_total)
            if generate and token_total else None,
        "mean_response_tokens": length_mean,
        "response_tokens_std":
            (sum((x["response_tokens"] - length_mean) ** 2 for x in examples) / len(examples)) ** 0.5
            if generate and examples else None,
        "mean_response_words": mean("response_words") if generate else None,
        "eval_batch_size": int(cfg.get("eval_batch_size", 1)),
        "generation_seed_scheme": "seed + batch_start_index",
    }


def evaluate_rows(bundle, rows=None, generate=True, initial_examples=None, progress_callback=None):
    cfg, tok, policy = bundle["cfg"], bundle["tokenizer"], bundle["policy"]
    rm, rm_tok = bundle["reward"]
    rows = bundle["rows"] if rows is None else rows
    device = next(policy.parameters()).device
    batch_size = max(1, int(cfg.get("eval_batch_size", 1)))

    examples = list(initial_examples or [])
    if len(examples) > len(rows):
        raise ValueError("Saved partial evaluation has more rows than the requested evaluation set")

    for i, item in enumerate(examples):
        expected = _source_id(rows[i], i)
        actual = str(item.get("source_index", i))
        if actual != expected:
            raise ValueError(
                f"Partial evaluation does not match the requested row order at index {i}: "
                f"{actual!r} != {expected!r}"
            )

    progress = tqdm(
        total=len(rows),
        initial=len(examples),
        desc="DPO evaluation",
        unit="example",
        dynamic_ncols=True,
    )

    try:
        for start in range(len(examples), len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            prompts = [prompt_messages_from_preference(row) for row in chunk]
            chosen_rejected = [preference_responses(row) for row in chunk]
            chosen = [pair[0] for pair in chosen_rejected]
            rejected = [pair[1] for pair in chosen_rejected]

            chosen_batch = _to_device(
                pad_batch(tok, [
                    encode_prompt_response(tok, prompt, response, int(cfg["max_sequence_length"]))
                    for prompt, response in zip(prompts, chosen)
                ]),
                device,
            )
            rejected_batch = _to_device(
                pad_batch(tok, [
                    encode_prompt_response(tok, prompt, response, int(cfg["max_sequence_length"]))
                    for prompt, response in zip(prompts, rejected)
                ]),
                device,
            )

            with torch.no_grad(), reference_mode(policy):
                ref_c = response_sequence_logprobs(policy, chosen_batch)[0].float()
                ref_r = response_sequence_logprobs(policy, rejected_batch)[0].float()

            with torch.no_grad():
                pol_c = response_sequence_logprobs(policy, chosen_batch)[0].float()
                pol_r = response_sequence_logprobs(policy, rejected_batch)[0].float()
                pair_rewards = score_reward_pairs(
                    rm,
                    rm_tok,
                    prompts + prompts,
                    chosen + rejected,
                ).float()

            batch_n = len(chunk)
            chosen_rewards = pair_rewards[:batch_n]
            rejected_rewards = pair_rewards[batch_n:]
            margins = (pol_c - ref_c) - (pol_r - ref_r)
            heldout_losses = -F.logsigmoid(float(cfg["beta"]) * margins)

            generated = None
            generated_rewards = None
            response_kls = None
            if generate:
                set_seed(int(cfg["seed"]) + start)
                generated = batch_generate(
                    policy,
                    tok,
                    prompts,
                    int(cfg["max_sequence_length"]),
                    int(cfg["max_generation_tokens"]),
                    **cfg["generation"],
                )

                with torch.no_grad():
                    generated_rewards = score_reward_pairs(
                        rm,
                        rm_tok,
                        prompts,
                        generated["responses"],
                    ).float()
                    policy_logp, _ = response_token_logprobs(
                        policy,
                        generated["sequences"],
                        generated["attention_mask"],
                        generated["prompt_width"],
                        generated["response_ids"],
                    )
                    with reference_mode(policy):
                        ref_logp, _ = response_token_logprobs(
                            policy,
                            generated["sequences"],
                            generated["attention_mask"],
                            generated["prompt_width"],
                            generated["response_ids"],
                        )

                mask = generated["response_mask"].to(policy_logp.dtype)
                numer = ((policy_logp - ref_logp) * mask).sum(-1)
                denom = mask.sum(-1).clamp_min(1.0)
                response_kls = numer / denom

            for j, row in enumerate(chunk):
                margin = float(margins[j])
                item = {
                    "source_index": row.get("source_index", start + j),
                    "policy_chosen_logp": float(pol_c[j]),
                    "policy_rejected_logp": float(pol_r[j]),
                    "reference_chosen_logp": float(ref_c[j]),
                    "reference_rejected_logp": float(ref_r[j]),
                    "chosen_reward": float(chosen_rewards[j]),
                    "rejected_reward": float(rejected_rewards[j]),
                    "preference_margin": margin,
                    "preference_correct": margin > 0,
                    "policy_only_preference_correct": float(pol_c[j]) > float(pol_r[j]),
                    "heldout_dpo_loss": float(heldout_losses[j]),
                    "chosen_words": word_count(chosen[j]),
                    "rejected_words": word_count(rejected[j]),
                }

                for key in (
                    "length_stratum",
                    "length_bucket",
                    "length_bin",
                    "stratum",
                    "bucket",
                    "category",
                    "chosen_length",
                    "rejected_length",
                ):
                    if key in row:
                        item[key] = row[key]

                if generate:
                    response = generated["responses"][j]
                    item.update({
                        "response": response,
                        "response_tokens": int(generated["response_lengths"][j]),
                        "response_words": word_count(response),
                        "response_reward": float(generated_rewards[j]),
                        "sampled_response_kl": float(response_kls[j]),
                    })

                examples.append(item)

            if progress_callback is not None:
                progress_callback(examples)
            progress.update(batch_n)
    finally:
        progress.close()

    return _summarize(examples, cfg, generate), examples


def run_evaluation(config_path, adapter, name, rows=None, beta=None, resume=False):
    bundle = load_evaluation_bundle(config_path, adapter)
    if beta is not None:
        bundle["cfg"] = {**bundle["cfg"], "beta": float(beta)}

    target_rows = bundle["rows"] if rows is None else rows
    out = repo_path(bundle["cfg"]["results_dir"])
    final_examples_path = out / f"{name}_examples.jsonl"
    partial_examples_path = out / f"{name}_examples.partial.jsonl"

    initial_examples = []
    if resume and partial_examples_path.exists():
        initial_examples = read_jsonl(partial_examples_path)
        print(
            f"[resume] Continuing {name} evaluation from "
            f"{len(initial_examples)}/{len(target_rows)} examples",
            flush=True,
        )

    set_seed(int(bundle["cfg"]["seed"]))
    metrics, examples = evaluate_rows(
        bundle,
        target_rows,
        initial_examples=initial_examples,
        progress_callback=lambda current: write_jsonl(partial_examples_path, current),
    )

    save_json(out / f"{name}_eval.json", metrics)
    write_jsonl(final_examples_path, examples)
    if partial_examples_path.exists():
        partial_examples_path.unlink()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()

    rows = None
    if args.max_examples is not None:
        cfg = load_yaml(args.config)
        rows = read_jsonl(cfg["paths"]["dpo_standard_eval"])[: int(args.max_examples)]

    print(run_evaluation(
        args.config,
        args.adapter,
        args.name,
        rows=rows,
        resume=args.resume,
    ))


if __name__ == "__main__":
    main()
