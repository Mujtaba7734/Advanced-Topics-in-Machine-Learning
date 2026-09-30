from __future__ import annotations

import argparse
from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from task1_dpo.train import run_training
from task1_dpo.evaluate import run_evaluation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = []
    for beta in cfg["betas"]:
        name = f"beta_{float(beta):.2f}"
        adapter = repo_path(cfg["standard_output"]).parent / name
        train = run_training(args.config, name, output_path=str(adapter), beta=float(beta),
                             max_examples=int(cfg["short_ablation_examples"]))
        evaluation = run_evaluation(args.config, str(adapter), name, beta=float(beta))
        results.append({"beta": float(beta), "train": train, "evaluation": evaluation})
    save_json(repo_path(cfg["results_dir"]) / "beta_study.json", results)


if __name__ == "__main__":
    main()
