from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    ids_path = outdir / "manual_audit_ids.csv"
    if ids_path.exists():
        previous = pd.read_csv(ids_path)
        if previous["xstest_id"].astype(int).tolist() != ids:
            raise ValueError("Existing fixed audit IDs disagree with configured seed/classes")
    else:
        pd.DataFrame({"xstest_id": ids}).to_csv(ids_path, index=False)
    sheet_path = outdir / "manual_audit_sheet.csv"
    existing_labels = {}
    if sheet_path.exists():
        prior = pd.read_csv(sheet_path, keep_default_na=False)
        if "policy" in prior.columns:
            existing_labels = {(int(r["xstest_id"]), str(r["policy"])): str(r["manual_label"])
                               for _, r in prior.iterrows()}
        else:
            existing_labels = {(int(r["xstest_id"]), policy): str(r[f"{policy}_manual_label"])
                               for _, r in prior.iterrows() for policy in ("sft", "dpo", "ppo", "grpo")}
    sheet = {sid: {"xstest_id": sid} for sid in ids}
    for policy in ("sft", "dpo", "ppo", "grpo"):
        rows = {int(r["xstest_id"]): r for r in read_jsonl(outdir / f"generated_{policy}.jsonl")}
        for sid in ids:
            row = rows[sid]
            sheet[sid].update({"benchmark_class": row["benchmark_class"], "type": row["type"],
                "prompt": row["prompt"], f"{policy}_response": row["response"],
                f"{policy}_manual_label": existing_labels.get((sid, policy), "")})
    pd.DataFrame([sheet[sid] for sid in ids]).to_csv(sheet_path, index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")
    print("Wrote unlabeled response sheet:", sheet_path)


if __name__ == "__main__":
    main()
