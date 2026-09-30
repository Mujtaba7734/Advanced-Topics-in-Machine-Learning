from __future__ import annotations

import argparse
import torch
from tqdm.auto import tqdm

from common.data import (encode_prompt_response, load_yaml, pad_batch, preference_responses,
                         prompt_messages_from_preference, read_jsonl, repo_path, write_jsonl)
from common.generation import (batch_generate, response_sequence_logprobs,
                               response_token_logprobs, score_reward_pairs)
from common.logging_utils import save_json, set_seed
from common.metrics import sampled_kl, word_count
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.dpo import dpo_loss


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate_rows(bundle, rows=None, generate=True):
    cfg, tok, policy = bundle["cfg"], bundle["tokenizer"], bundle["policy"]
    rm, rm_tok = bundle["reward"]
    rows = bundle["rows"] if rows is None else rows
    device = next(policy.parameters()).device
    examples = []
    for row in tqdm(rows, total=len(rows), desc="DPO evaluation", unit="example", dynamic_ncols=True):
        prompt = prompt_messages_from_preference(row)
        chosen, rejected = preference_responses(row)
        batches = [pad_batch(tok, [encode_prompt_response(tok, prompt, response,
                    int(cfg["max_sequence_length"]))]) for response in (chosen, rejected)]
        batches = [{k: v.to(device) for k, v in batch.items()} for batch in batches]
        with torch.no_grad(), reference_mode(policy):
            ref = [float(response_sequence_logprobs(policy, b)[0][0]) for b in batches]
        with torch.no_grad():
            pol = [float(response_sequence_logprobs(policy, b)[0][0]) for b in batches]
            rewards = score_reward_pairs(rm, rm_tok, [prompt, prompt], [chosen, rejected])
        margin = (pol[0] - ref[0]) - (pol[1] - ref[1])
        heldout_loss, _ = dpo_loss(*(torch.tensor([v]) for v in (pol[0], pol[1], ref[0], ref[1])),
                                   float(cfg["beta"]))
        item = {"source_index": row.get("source_index"), "policy_chosen_logp": pol[0],
                "policy_rejected_logp": pol[1], "reference_chosen_logp": ref[0],
                "reference_rejected_logp": ref[1], "chosen_reward": float(rewards[0]),
                "rejected_reward": float(rewards[1]), "preference_margin": margin,
                "preference_correct": margin > 0,
                "policy_only_preference_correct": pol[0] > pol[1],
                "heldout_dpo_loss": float(heldout_loss),
                "chosen_words": word_count(chosen), "rejected_words": word_count(rejected)}
        for key in ("length_stratum", "length_bucket", "length_bin", "stratum", "bucket",
                    "category", "chosen_length", "rejected_length"):
            if key in row:
                item[key] = row[key]
        if generate:
            gen = batch_generate(policy, tok, [prompt], int(cfg["max_sequence_length"]),
                int(cfg["max_generation_tokens"]), **cfg["generation"])
            response = gen["responses"][0]
            with torch.no_grad():
                reward = score_reward_pairs(rm, rm_tok, [prompt], [response])
                sequences, attention, response_ids = (gen["sequences"], gen["attention_mask"],
                                                      gen["response_ids"])
                policy_logp, _ = response_token_logprobs(policy, sequences, attention,
                                                          gen["prompt_width"], response_ids)
                with reference_mode(policy):
                    ref_logp, _ = response_token_logprobs(policy, sequences, attention,
                                                           gen["prompt_width"], response_ids)
                response_kl = sampled_kl(policy_logp, ref_logp, gen["response_mask"])
            item.update({"response": response, "response_tokens": gen["response_lengths"][0],
                         "response_words": word_count(response), "response_reward": float(reward[0]),
                         "sampled_response_kl": float(response_kl)})
        examples.append(item)
    def mean(key):
        return sum(float(x[key]) for x in examples) / len(examples) if examples else None
    length_mean = mean("response_tokens") if generate else None
    metrics = {"n": len(examples), "preference_accuracy": mean("preference_correct"),
        "reference_adjusted_accuracy": mean("preference_correct"),
        "policy_only_preference_accuracy": mean("policy_only_preference_correct"),
        "heldout_dpo_loss": mean("heldout_dpo_loss"), "beta": float(cfg["beta"]),
        "mean_reference_log_ratio_chosen": (sum(x["policy_chosen_logp"] - x["reference_chosen_logp"] for x in examples) / len(examples)) if examples else None,
        "mean_reward_margin": (sum(x["chosen_reward"] - x["rejected_reward"] for x in examples) / len(examples)) if examples else None,
        "mean_response_reward": mean("response_reward") if generate else None,
        "sampled_response_kl": (sum(x["sampled_response_kl"] * x["response_tokens"] for x in examples) /
                                sum(x["response_tokens"] for x in examples))
                               if generate and sum(x["response_tokens"] for x in examples) else None,
        "mean_response_tokens": length_mean,
        "response_tokens_std": (sum((x["response_tokens"] - length_mean) ** 2 for x in examples) /
                                len(examples)) ** 0.5 if generate and examples else None,
        "mean_response_words": mean("response_words") if generate else None}
    return metrics, examples


def run_evaluation(config_path, adapter, name, rows=None, beta=None):
    bundle = load_evaluation_bundle(config_path, adapter)
    if beta is not None:
        bundle["cfg"] = {**bundle["cfg"], "beta": float(beta)}
    set_seed(int(bundle["cfg"]["seed"]))
    metrics, examples = evaluate_rows(bundle, rows)
    out = repo_path(bundle["cfg"]["results_dir"])
    save_json(out / f"{name}_eval.json", metrics)
    write_jsonl(out / f"{name}_examples.jsonl", examples)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    print(run_evaluation(args.config, args.adapter, args.name))


if __name__ == "__main__":
    main()
