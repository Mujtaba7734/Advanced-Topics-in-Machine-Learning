from __future__ import annotations

import argparse
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from common.generation import response_sequence_logprobs
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    model, optimizer = bundle["model"], bundle["optimizer"]
    accum = int(cfg["grad_accum_steps"])
    log_path = repo_path(cfg["results_dir"]) / f"{run_name}_train.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    optimizer.zero_grad(set_to_none=True)
    timer = wall_timer()
    steps = 0
    for epoch in range(int(cfg["epochs"])):
        loader = bundle["loader"]
        for i, (chosen, rejected) in enumerate(loader):
            device = next(model.parameters()).device
            chosen = {k: v.to(device) for k, v in chosen.items()}
            rejected = {k: v.to(device) for k, v in rejected.items()}
            with torch.no_grad(), reference_mode(model):
                ref_c, _, _ = response_sequence_logprobs(model, chosen)
                ref_r, _, _ = response_sequence_logprobs(model, rejected)
            pol_c, _, _ = response_sequence_logprobs(model, chosen)
            pol_r, _, _ = response_sequence_logprobs(model, rejected)
            loss, diagnostics = dpo_loss(pol_c, pol_r, ref_c, ref_r, bundle["beta"])
            # Scale the final partial accumulation window by its actual size.
            window = min(accum, len(loader) - (i // accum) * accum)
            (loss / window).backward()
            if (i + 1) % accum == 0 or i + 1 == len(loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), float(cfg["max_grad_norm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                append_jsonl(log_path, {"epoch": epoch, "step": steps, "loss": float(loss.item()),
                    "grad_norm": float(grad_norm), "beta": bundle["beta"],
                    **{k: float(v) for k, v in diagnostics.items()}})
    model.save_pretrained(output)
    summary = {"run_name": run_name, "adapter": str(output), "examples": len(bundle["rows"]),
               "epochs": int(cfg["epochs"]), "optimizer_steps": steps, "beta": bundle["beta"],
               "wall_seconds": timer()}
    save_json(repo_path(cfg["results_dir"]) / f"{run_name}_config.json",
              {**cfg, "beta": bundle["beta"], "training_data": dataset_path or cfg["paths"]["dpo_standard_train"],
               "max_examples": max_examples})
    save_json(repo_path(cfg["results_dir"]) / f"{run_name}_train.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
