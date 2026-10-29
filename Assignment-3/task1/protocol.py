"""Fixed Task 1 experiment settings from ATML-PA3.pdf."""
import hashlib
import json


FIXED = {
    "seed": 6304, "train_examples": 40000, "epochs": 1,
    "max_length": 2048, "microbatch": 2, "grad_accum": 16,
    "optimizer_updates": 1250, "warmup_updates": 38,
    "weight_decay": 0.01, "adam_betas": [0.9, 0.999],
    "adam_eps": 1e-8, "max_grad_norm": 1.0,
    "fp16": True, "gradient_checkpointing": True,
    "full_lr": 2e-5, "adapter_lr": 2e-4,
    "l2sp_lambda": 1e-3, "lora_dropout": 0.05,
    "op_rank": 128, "checkpoint_updates": [312, 625, 937, 1250],
    "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "diagnostic_layers": [0, 13, 27],
    "gsm8k_max_new_tokens": 1024, "qualitative_max_new_tokens": 512,
}


def validate_protocol(config):
    for key, expected in FIXED.items():
        if config.get(key) != expected:
            raise ValueError(f"Task 1 protocol mismatch for {key}: expected {expected!r}, got {config.get(key)!r}")
    if config["microbatch"] * config["grad_accum"] * config["optimizer_updates"] != config["train_examples"]:
        raise ValueError("Examples, microbatch, accumulation, and update counts disagree")


def config_sha256(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
