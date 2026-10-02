"""Shared frozen held-out protocol for PPO and GRPO policies."""
from __future__ import annotations

import torch
from tqdm.auto import tqdm

from common.data import prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import masked_mean, sample_entropy
from common.models import reference_mode


def _row_id(row: dict, position: int):
    return row.get("source_index", position)


def _validate_resume_prefix(existing: list[dict], rows: list[dict]) -> None:
    if len(existing) > len(rows):
        raise ValueError(
            f"Saved evaluation has {len(existing)} rows but fixed evaluation set "
            f"has only {len(rows)}"
        )
    for i, saved in enumerate(existing):
        expected = _row_id(rows[i], i)
        if str(saved.get("source_index")) != str(expected):
            raise ValueError(
                "Saved evaluation is not a prefix of the fixed evaluation set: "
                f"row {i} has source_index={saved.get('source_index')!r}, "
                f"expected {expected!r}"
            )


def evaluate_rl(bundle, name, task, resume: bool = False):
    cfg, tok, policy = bundle["cfg"], bundle["tokenizer"], bundle["policy"]
    rm, rm_tok = bundle["reward"]
    rows = bundle["rows"]

    max_response = int(
        cfg.get(
            "eval_max_response_length",
            cfg.get("max_completion_length"),
        )
    )
    batch_size = int(cfg.get("eval_batch_size", 1))
    if batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")

    seed_by_batch = bool(cfg.get("eval_seed_by_batch", False))
    out = repo_path(cfg["results_dir"])
    examples_path = out / f"{name}_examples.jsonl"

    examples: list[dict] = []
    start = 0

    if resume and examples_path.exists():
        existing = read_jsonl(examples_path)
        _validate_resume_prefix(existing, rows)

        # A batch is the reproducibility unit when eval_seed_by_batch is enabled.
        # If a session died while writing one batch, discard that partial batch
        # and regenerate it from the same batch seed.
        if len(existing) == len(rows):
            keep = len(existing)
        else:
            keep = (len(existing) // batch_size) * batch_size

        examples = existing[:keep]
        if keep != len(existing):
            write_jsonl(examples_path, examples)
            print(
                f"[resume] Dropped {len(existing) - keep} partial evaluation rows "
                f"to restart the interrupted batch at index {keep}",
                flush=True,
            )
        start = keep

        if start:
            print(
                f"[resume] Reusing {start}/{len(rows)} completed "
                f"{task.upper()} evaluation examples for {name}",
                flush=True,
            )
    else:
        write_jsonl(examples_path, [])

    if not seed_by_batch:
        if resume and start:
            raise RuntimeError(
                "Safe mid-evaluation resume requires eval_seed_by_batch=true "
                "so remaining sampled responses reproduce independently of "
                "prior RNG state."
            )
        set_seed(int(cfg["seed"]))

    batch_starts = list(range(start, len(rows), batch_size))
    progress = tqdm(
        batch_starts,
        total=len(batch_starts),
        desc=f"{task.upper()} evaluation: {name}",
        unit="batch",
        dynamic_ncols=True,
    )

    for batch_start in progress:
        batch_rows = rows[batch_start : batch_start + batch_size]
        if seed_by_batch:
            set_seed(int(cfg["seed"]) + batch_start)

        messages = [prompt_messages(row) for row in batch_rows]
        gen = batch_generate(
            policy,
            tok,
            messages,
            int(cfg["max_prompt_length"]),
            max_response,
            **cfg["generation"],
        )
        seq = gen["sequences"]
        attn = gen["attention_mask"]
        response_ids = gen["response_ids"]
        mask = gen["response_mask"]

        with torch.no_grad():
            logp, _ = response_token_logprobs(
                policy,
                seq,
                attn,
                gen["prompt_width"],
                response_ids,
            )
            with reference_mode(policy):
                ref, _ = response_token_logprobs(
                    policy,
                    seq,
                    attn,
                    gen["prompt_width"],
                    response_ids,
                )
            reward = score_reward_pairs(
                rm,
                rm_tok,
                messages,
                gen["responses"],
                int(
                    cfg.get(
                        "reward_max_length",
                        int(cfg["max_prompt_length"]) + max_response,
                    )
                ),
            )

        if not (
            torch.isfinite(logp).all()
            and torch.isfinite(ref).all()
            and torch.isfinite(reward).all()
        ):
            raise FloatingPointError(
                f"Non-finite held-out evaluation tensor in {name} "
                f"at batch index {batch_start}"
            )

        batch_examples = []
        for local_i, row in enumerate(batch_rows):
            row_mask = mask[local_i : local_i + 1]
            row_logp = logp[local_i : local_i + 1]
            row_ref = ref[local_i : local_i + 1]
            global_i = batch_start + local_i

            record = {
                "source_index": _row_id(row, global_i),
                "response": gen["responses"][local_i],
                "reward": float(reward[local_i]),
                "sampled_kl": float(
                    masked_mean(row_logp - row_ref, row_mask)
                ),
                "entropy": float(sample_entropy(row_logp, row_mask)),
                "response_tokens": int(gen["response_lengths"][local_i]),
                "terminated_with_eos": bool(
                    gen["terminated_with_eos"][local_i]
                ),
                "truncated": bool(gen["truncated"][local_i]),
            }
            batch_examples.append(record)

        # Persist each finished batch immediately. A Kaggle timeout therefore
        # loses at most one batch, not the full held-out evaluation.
        for record in batch_examples:
            append_jsonl(examples_path, record)
        examples.extend(batch_examples)

    def mean(key):
        return (
            sum(float(row[key]) for row in examples) / len(examples)
            if examples
            else None
        )

    metrics = {
        "n": len(examples),
        "mean_reward": mean("reward"),
        "mean_sampled_kl": mean("sampled_kl"),
        "mean_entropy": mean("entropy"),
        "mean_response_tokens": mean("response_tokens"),
        "eos_rate": mean("terminated_with_eos"),
        "truncation_rate": mean("truncated"),
        "eval_batch_size": batch_size,
        "generation_seed_scheme": (
            "seed + batch_start_index"
            if seed_by_batch
            else "single global seed"
        ),
    }
    save_json(out / f"{name}_eval.json", metrics)
    return metrics
