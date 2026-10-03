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

    exact_accuracy_drop = {
        name:
            gsm["policies"][name]["exact_accuracy"]
            - transfer["policies"][name]["exact_accuracy"]
        for name in ("sft", "rlvr", "rlaif")
    }
    response_length_shift = {
        name:
            transfer["policies"][name]["mean_response_tokens"]
            - gsm["policies"][name]["mean_response_tokens"]
        for name in ("sft", "rlvr", "rlaif")
    }
    format_compliance_drop = {
        name:
            gsm["policies"][name]["format_compliance"]
            - transfer["policies"][name]["format_compliance"]
        for name in ("sft", "rlvr", "rlaif")
    }

    pairwise_score_drop = {}
    for pair_name, gsm_pair in gsm["pairwise"].items():
        transfer_pair = transfer["pairwise"][pair_name]
        pairwise_score_drop[pair_name] = {
            "a_pairwise_score_drop":
                gsm_pair["a_pairwise_score"]
                - transfer_pair["a_pairwise_score"],
            "b_pairwise_score_drop":
                gsm_pair["b_pairwise_score"]
                - transfer_pair["b_pairwise_score"],
        }

    comparison = {
        "in_domain": gsm,
        "transfer": transfer,
        "perturbations": perturb,
        "exact_accuracy_drop_gsm_minus_transfer": exact_accuracy_drop,
        "response_length_shift_transfer_minus_gsm": response_length_shift,
        "format_compliance_drop_gsm_minus_transfer": format_compliance_drop,
        "pairwise_score_drop_gsm_minus_transfer": pairwise_score_drop,
        "rlvr_minus_rlaif_exact_accuracy": {
            "gsm":
                gsm["policies"]["rlvr"]["exact_accuracy"]
                - gsm["policies"]["rlaif"]["exact_accuracy"],
            "transfer":
                transfer["policies"]["rlvr"]["exact_accuracy"]
                - transfer["policies"]["rlaif"]["exact_accuracy"],
        },
    }

    save_json(outdir / "feedback_comparison.json", comparison)
    print(outdir / "feedback_comparison.json")


if __name__ == "__main__":
    main()
