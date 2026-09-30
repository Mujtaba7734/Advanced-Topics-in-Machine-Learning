from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl
from common.models import load_policy, load_reward_model, load_tokenizer
from common.rl_evaluation import evaluate_rl


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    print(evaluate_rl(load_evaluation_bundle(args.config, args.adapter), args.name, "grpo"))


if __name__ == "__main__":
    main()
