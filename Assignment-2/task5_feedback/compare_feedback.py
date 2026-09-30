from __future__ import annotations

import argparse
from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    gsm = load_json(outdir / "gsm_metrics.json")
    transfer = load_json(outdir / "transfer_metrics.json")
    perturb = load_json(outdir / "perturbation_metrics.json")
    comparison = {"in_domain": gsm, "transfer": transfer, "perturbations": perturb,
        "accuracy_shift": {name: transfer["policies"][name]["exact_accuracy"] - gsm["policies"][name]["exact_accuracy"]
                           for name in ("sft", "rlvr", "rlaif")},
        "rlvr_minus_rlaif": {dataset: data["policies"]["rlvr"]["exact_accuracy"] -
                            data["policies"]["rlaif"]["exact_accuracy"]
                            for dataset, data in (("gsm", gsm), ("transfer", transfer))}}
    save_json(outdir / "feedback_comparison.json", comparison)
    print(outdir / "feedback_comparison.json")


if __name__ == "__main__":
    main()
