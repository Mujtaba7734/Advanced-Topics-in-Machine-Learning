"""Rebuild the small report metrics and PPO clip plot from verified task backups.

Run from Assignment-2/Report: python scripts/rebuild_missing_evidence.py
This script reads backups and writes only aggregate report assets.
"""
from __future__ import annotations

import csv
import hashlib
import json
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parent
MANIFEST = json.loads((REPORT / "data/evidence_manifest.json").read_text(encoding="utf-8"))
BACKUPS = {
    "1": ROOT / "backups/ATML_PA2_Task1_BACKUP (1).tar.gz",
    "2": ROOT / "backups/ATML_PA2_Task2_BACKUP (1).tar.gz",
    "3": ROOT / "backups/ATML_PA2_Task3_BACKUP (1).tar.gz",
    "5": ROOT / "backups/ATML_PA2_Task5_BACKUP.tar.gz",
}


def checked_archive(task):
    path = BACKUPS[task]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = MANIFEST["archival_sources"][task]["sha256"]
    if digest != expected:
        raise ValueError(f"Task {task} archive checksum mismatch: {path}")
    return tarfile.open(path, "r:gz")


def load_json(archive, name):
    member = archive.extractfile(name)
    if member is None:
        raise ValueError(f"Missing archive member: {name}")
    return json.load(member)


def load_jsonl(archive, name):
    member = archive.extractfile(name)
    if member is None:
        raise ValueError(f"Missing archive member: {name}")
    return [json.loads(line) for line in member if line.strip()]


def write_csv(name, headers, rows):
    with (REPORT / "data" / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def main():
    with checked_archive("1") as archive:
        study = load_json(archive, "results/task1_dpo/length_study.json")
    dpo_rows = []
    for policy in ("standard", "length_balanced"):
        metrics = study["stratified"][policy]
        word = study["word_limits"][policy]
        dpo_rows.append([
            policy, metrics["n"], metrics["mean_response_tokens"],
            metrics["response_tokens_std"], word["n"], word["compliance_rate"] * 100,
        ])
    write_csv(
        "03_dpo_wordlimit.csv",
        ["Condition", "StratifiedEvalN", "StratifiedMeanTok", "StratifiedSDTok", "WordLimitN", "CompliancePct"],
        dpo_rows,
    )

    with checked_archive("2") as archive:
        clipping = load_json(archive, "results/task2_ppo/clipping_study.json")
        trajectory = load_jsonl(archive, "results/task2_ppo/standard_train.jsonl")
        ppo_compute = load_json(archive, "results/task2_ppo/standard_train.json")
    with checked_archive("3") as archive:
        grpo_compute = load_json(archive, "results/task3_grpo/standard_train.json")
    write_csv(
        "20_training_compute.csv",
        ["Method", "Updates", "WallSeconds", "PeakVRAMBytes", "PeakVRAMGiB"],
        [[method, metrics["updates"], metrics["wall_seconds"], metrics["peak_vram_bytes"],
          metrics["peak_vram_bytes"] / (1024 ** 3)]
         for method, metrics in (("PPO", ppo_compute), ("GRPO", grpo_compute))],
    )
    if [row["update"] for row in trajectory] != list(range(1, 21)):
        raise ValueError("Expected the fixed 20-update PPO continuation")
    forks = {float(row["clip_epsilon"]): row for row in clipping["forks"]}
    clip_rows = []
    for eps in (0.05, 0.20, 0.50):
        cached = clipping["cached"][str(eps)]
        fork = forks[eps]
        evaluation = fork["evaluation"]
        clip_rows.append([
            eps, cached["affected_token_fraction"] * 100, cached["mean_surrogate"],
            evaluation["mean_reward"], evaluation["mean_sampled_kl"],
            evaluation["mean_response_tokens"], fork["stability"]["policy_grad_norm_std"],
        ])
    write_csv(
        "04_ppo_clip.csv",
        ["Epsilon", "CachedAffectedPct", "CachedSurrogate", "HeldoutReward", "HeldoutKL", "MeanTok", "PolicyGradNormSD"],
        clip_rows,
    )
    write_csv(
        "16_ppo_clip_trajectory.csv",
        ["Update", "TrainingClipFraction"],
        [[row["update"], row["clip_fraction"]] for row in trajectory],
    )
    fig, ax = plt.subplots(figsize=(5.2, 1.8))
    ax.plot([row["update"] for row in trajectory], [row["clip_fraction"] for row in trajectory], marker="o", markersize=2)
    ax.set(xlabel="PPO continuation update", ylabel="Clip fraction", xlim=(1, 20), ylim=(-0.002, 0.02))
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORT / "figures/26_ppo_clip_trajectory.png", dpi=200)
    plt.close(fig)

    with checked_archive("5") as archive:
        comparison = load_json(archive, "results/task5_feedback/feedback_comparison.json")
        diagnostic = load_json(archive, "results/task5_feedback/perturbation_metrics.json")
    transfer_rows = []
    for policy in ("sft", "rlvr", "rlaif"):
        transfer_rows.append([
            policy,
            100 * comparison["exact_accuracy_drop_gsm_minus_transfer"][policy],
            100 * comparison["format_compliance_drop_gsm_minus_transfer"][policy],
            comparison["response_length_shift_transfer_minus_gsm"][policy],
        ])
    write_csv(
        "17_task5_transfer.csv",
        ["Policy", "AccuracyGsmMinusSvampPp", "FormatGsmMinusSvampPp", "TokensSvampMinusGsm"],
        transfer_rows,
    )
    pair_rows = []
    for pair, values in comparison["pairwise_score_drop_gsm_minus_transfer"].items():
        pair_rows.append([pair, 100 * values["a_pairwise_score_drop"], 100 * values["b_pairwise_score_drop"]])
    write_csv(
        "18_task5_pairwise_transfer.csv",
        ["Pair", "AScoreGsmMinusSvampPp", "BScoreGsmMinusSvampPp"],
        pair_rows,
    )
    write_csv(
        "19_task5_sensitivity.csv",
        ["Metric", "RLVRPct", "RLAIFPct"],
        [[name, 100 * diagnostic[name]["rlvr"], 100 * diagnostic[name]["rlaif"]]
         for name in ("S_reason", "S_outcome")],
    )
    print("Rebuilt DPO length, PPO/GRPO compute, PPO clipping, and Task 5 transfer/sensitivity evidence.")


if __name__ == "__main__":
    main()
