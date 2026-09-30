from __future__ import annotations

import argparse
from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import load_evaluation_bundle
from common.rl_evaluation import evaluate_rl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = []
    for beta in cfg["kl_values"]:
        name = f"kl_{float(beta):.2f}"
        adapter = repo_path(cfg["output"]).parent / name
        train = run_ppo(args.config, str(adapter), int(cfg["fork_updates"]),
                        kl_beta=float(beta), run_name=name)
        evaluation = evaluate_rl(load_evaluation_bundle(args.config, str(adapter)), name, "ppo")
        results.append({"kl_beta": float(beta), "train": train, "evaluation": evaluation})
    save_json(repo_path(cfg["results_dir"]) / "kl_study.json", results)


if __name__ == "__main__":
    main()
