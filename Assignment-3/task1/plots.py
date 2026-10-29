import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from task1.evaluation import CONDITIONS, TASKS


PRIMARY = {"gsm8k_cot": "exact_match,flexible-extract", "hellaswag": "acc_norm,none",
           "boolq": "acc,none", "arc_easy": "acc_norm,none"}


def aggregate(config):
    root = Path(config["result_root"])
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    benchmark_rows = []
    for condition in CONDITIONS:
        for task in TASKS:
            path = root / "evaluation" / condition / f"{task}.json"
            if not path.exists():
                benchmark_rows.append({"condition": condition, "task": task, "status": "missing",
                                       "primary": None, "raw_accuracy": None, "normalized_accuracy": None})
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("sample_count") != TASKS[task]:
                benchmark_rows.append({"condition": condition, "task": task, "status": "incomplete",
                                       "primary": None, "raw_accuracy": None, "normalized_accuracy": None})
                continue
            scores = record["results"]
            key = PRIMARY[task]
            if key not in scores:
                raise KeyError(f"Primary metric {key} missing for {task}: {list(scores)}")
            benchmark_rows.append({"condition": condition, "task": task, "status": "complete",
                                   "primary": float(scores[key]),
                                   "raw_accuracy": float(scores["acc,none"]) if "acc,none" in scores else None,
                                   "normalized_accuracy": float(scores["acc_norm,none"]) if "acc_norm,none" in scores else None})
    benchmark = pd.DataFrame(benchmark_rows)
    baseline = benchmark[(benchmark["condition"] == "baseline") & benchmark["primary"].notna()].set_index("task")["primary"]
    benchmark["change_pp"] = benchmark.apply(
        lambda row: 100 * (row["primary"] - baseline[row["task"]])
        if pd.notna(row["primary"]) and row["task"] in baseline else None, axis=1)
    benchmark.to_csv(out / "benchmarks.csv", index=False)
    wide_score = benchmark.pivot(index="condition", columns="task", values="primary")
    wide_change = benchmark.pivot(index="condition", columns="task", values="change_pp")
    wide_score.columns = [f"{name}_score" for name in wide_score.columns]
    wide_change.columns = [f"{name}_change_pp" for name in wide_change.columns]
    pd.concat([wide_score, wide_change], axis=1).reindex(CONDITIONS).to_csv(out / "benchmarks_wide.csv")
    if benchmark["primary"].notna().any():
        pivot = benchmark.pivot(index="condition", columns="task", values="primary")
        pivot.plot.bar(figsize=(12, 5), rot=45)
        plt.tight_layout(); plt.savefig(out / "benchmarks.png", dpi=160); plt.close()
    logs, resources = [], []
    for method in ("full", "l2sp", "lora4", "lora16", "oplora16"):
        path = Path(config["output_root"]) / method / "train.jsonl"
        if not path.exists():
            resources.append({"method": method, "status": "missing"})
            continue
        if path.stat().st_size == 0:
            resources.append({"method": method, "status": "incomplete"})
            continue
        data = pd.read_json(path, lines=True)
        if data.empty:
            resources.append({"method": method, "status": "incomplete"})
            continue
        data["method"] = method
        logs.append(data)
        last = data.iloc[-1]
        ckpt = Path(config["output_root"]) / method / "inference_1250.pt"
        recovery = Path(config["output_root"]) / method / "recovery_1250.pt"
        metadata_path = Path(config["output_root"]) / method / "model_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        resources.append({"method": method, "status": "complete" if last["update"] == 1250 and ckpt.exists() else "incomplete",
                          "training_seconds": last.training_seconds, "peak_gpu_bytes": data.peak_gpu_bytes.max(),
                          "checkpoint_bytes": ckpt.stat().st_size if ckpt.exists() else None,
                          "recovery_bytes": recovery.stat().st_size if recovery.exists() else None,
                          "trainable_parameters": _trainable_from_checkpoint(config, method),
                          "projected_checkpoint_bytes": _projected_bytes(config, method),
                          "basis_bytes": metadata.get("basis_bytes"),
                          "basis_seconds": metadata.get("basis_seconds"),
                          "reference_bytes": last.reference_bytes,
                          "reference_transfer_seconds": last.reference_transfer_seconds})
    if logs:
        combined = pd.concat(logs)
        combined.to_csv(out / "training_log.csv", index=False)
        for method, group in combined.groupby("method"):
            plt.plot(group["update"], group["sft_loss"], label=method)
        plt.xlabel("Optimizer update"); plt.ylabel("Response-token SFT loss")
        plt.legend(); plt.tight_layout(); plt.savefig(out / "sft_loss.png", dpi=160); plt.close()
        l2 = combined[combined["method"] == "l2sp"]
        if not l2.empty:
            l2.plot(x="update", y=["omega", "penalty", "total_loss"])
            plt.tight_layout(); plt.savefig(out / "l2sp.png", dpi=160); plt.close()
        displacement = combined[combined["update"].isin([625, 1250]) & combined["method"].isin(["full", "l2sp"])]
        if "parameter_displacement_l2" in displacement.columns and not displacement.empty:
            displacement.pivot(index="update", columns="method", values="parameter_displacement_l2").plot.bar()
            plt.ylabel("Full parameter displacement (L2)")
            plt.tight_layout(); plt.savefig(out / "parameter_displacement.png", dpi=160); plt.close()
    pd.DataFrame(resources).to_csv(out / "resources.csv", index=False)
    diagnostic_root = root / "diagnostics"
    for name, columns, figure in (("weight_space.csv", ["rho", "eta"], "weight_space.png"),
                                   ("activation_ratios.csv", ["ratio"], "activation_ratios.png")):
        path = diagnostic_root / name
        if path.exists():
            data = pd.read_csv(path)
            if not data.empty:
                group_columns = ["condition", "layer"] + (["group"] if "group" in data.columns else [])
                data.groupby(group_columns)[columns].mean().plot.bar(figsize=(14, 5))
                plt.tight_layout(); plt.savefig(out / figure, dpi=160); plt.close()
    return out


def _trainable_from_checkpoint(config, method):
    path = Path(config["output_root"]) / method / "model_metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("trainable_parameters")


def _projected_bytes(config, method):
    path = Path(config["output_root"]) / method / "projected_1250.pt"
    return path.stat().st_size if path.exists() else None
