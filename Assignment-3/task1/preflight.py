"""Real-interface checks run explicitly on Kaggle before any training run."""
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
import tempfile
from pathlib import Path

import torch

from common.checkpoint import atomic_json
from task1.dataset import load_selected
from task1.model import load_model
from task1.prepare_data import encode_example
from task1.protocol import config_sha256, validate_protocol


def source_sha256():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "task1").glob("*.py")) + sorted((root / "common").glob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def dependency_versions(requirements_path):
    from packaging.version import Version
    versions = {}
    for line in Path(requirements_path).read_text(encoding="utf-8").splitlines():
        if "==" not in line or line.startswith("#"):
            continue
        name, required = line.split("==", 1)
        actual = importlib.metadata.version(name)
        versions[name] = {"required": required, "actual": actual}
        if Version(actual).public != Version(required).public:
            raise RuntimeError(f"Dependency mismatch: {name} requires {required}, installed {actual}")
    return versions


def resource_snapshot(config):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; enable the Kaggle GPU accelerator")
    devices = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        devices.append({"index": index, "name": torch.cuda.get_device_name(index),
                        "free_bytes": free, "total_bytes": total})
    if os.name == "posix":
        ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        available_kib = next((int(line.split()[1]) for line in meminfo.splitlines()
                              if line.startswith("MemAvailable:")), None)
        ram_available_bytes = available_kib * 1024 if available_kib is not None else None
    else:
        ram_bytes = None
        ram_available_bytes = None
    return {"devices": devices, "ram_bytes": ram_bytes,
            "ram_available_bytes": ram_available_bytes,
            "disk_free_bytes": shutil.disk_usage(config["output_root"]).free,
            "cuda_runtime": torch.version.cuda}


def inspect_tokenizer(config, output_dir):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"], cache_dir=config["cache_dir"],
                                               revision=config.get("model_revision"))
    cases = [
        ("What is 2 + 2?", "4"),
        ("Solve $x^2-4=0$.", "The roots are $x=\\pm2$."),
        ("Use symbols: π, \"quotes\", <>& and \n newlines.", "  Answer: π ≈ 3.14159.\n"),
        ("Token boundary check", "\n  leading and trailing spaces  "),
        ("Long response", "Math step. " * 900),
    ]
    records = []
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    for question, answer in cases:
        encoded = encode_example(tokenizer, question, answer)
        p = encoded["prompt_tokens"]
        if any(label != -100 for label in encoded["labels"][:p]):
            raise AssertionError("Prompt is not fully masked")
        if encoded["labels"][p:] != encoded["input_ids"][p:]:
            raise AssertionError("Response labels differ from response tokens")
        if encoded["labels"][-1] != end_id:
            raise AssertionError("Final supervised token is not <|im_end|>")
        decoded_prompt = tokenizer.decode(encoded["input_ids"][:p], skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False)
        if decoded_prompt.count("<|im_start|>assistant") != 1:
            raise AssertionError("Duplicate or missing assistant header")
        if decoded_prompt.count("<think>\n\n</think>\n\n") != 1:
            raise AssertionError("Qwen non-thinking assistant prefix is missing or duplicated")
        records.append({"question": question, "answer": answer[:120],
                        "sequence_length": len(encoded["input_ids"]),
                        "prompt_tokens": p, "response_tokens": encoded["response_tokens"],
                        "decoded_prompt_tail": decoded_prompt[-120:],
                        "decoded_target": tokenizer.decode(encoded["input_ids"][p:],
                                                           skip_special_tokens=False,
                                                           clean_up_tokenization_spaces=False)[:160]})
    if records[-1]["sequence_length"] <= config["max_length"]:
        raise AssertionError("Long-answer case failed to exercise the 2,048-token limit")
    atomic_json(records, output_dir / "tokenization.json", indent=2, ensure_ascii=False)
    return tokenizer


def inspect_data(config, tokenizer):
    from task1.data_source import load_split
    rows, manifest = load_selected(config)
    if not Path(config["model"]).exists() and not manifest.get("model_revision"):
        raise AssertionError("Hugging Face model revision was not pinned during preparation")
    if len(rows) != 40000 or rows[-1]["source_index"] <= rows[0]["source_index"]:
        raise AssertionError("Prepared selection count or order is wrong")
    if any(len(row["input_ids"]) > 2048 or row["response_tokens"] < 2 for row in rows):
        raise AssertionError("Invalid selected sequence length or response")
    if any(row["labels"][-1] != tokenizer.convert_tokens_to_ids("<|im_end|>") for row in rows):
        raise AssertionError("Selected examples lack supervised end-of-turn tokens")
    source = load_split(config["metamath_dataset"], "train", config["cache_dir"])
    gsm = load_split(config["gsm8k_dataset"], "test", config["cache_dir"], "main")
    if not {"query", "response", "original_question"}.issubset(source.column_names):
        raise AssertionError(f"MetaMathQA columns: {source.column_names}")
    if len(gsm) != 1319 or "question" not in gsm.column_names:
        raise AssertionError("GSM8K test split or schema mismatch")
    if source._fingerprint != manifest["source_fingerprint"] or gsm._fingerprint != manifest["gsm8k_fingerprint"]:
        raise AssertionError("Dataset fingerprint changed since preparation")
    for row in (rows[0], rows[len(rows)//2], rows[-1]):
        original = source[row["source_index"]]
        expected = encode_example(tokenizer, original["query"], original["response"])
        if row["input_ids"] != expected["input_ids"] or row["labels"] != expected["labels"]:
            raise AssertionError(f"Prepared tokenization differs at source row {row['source_index']}")
    return rows, manifest


from task1.preflight_memory import inspect_model_method


def inspect_harness_and_qualitative(config, tokenizer):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from task1.evaluation import select_task_samples, normalized_choice_prediction
    from task1.plots import PRIMARY
    from task1.qualitative import load_prompts
    model, _ = load_model(config, "full", for_inference=True)
    model.eval(); model.config.use_cache = True
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1, enable_thinking=False)
    samples_by_task = {}
    for task in ("gsm8k_cot", "hellaswag", "boolq", "arc_easy"):
        kwargs = {"model": lm, "tasks": [task], "samples": {task: [0, 1]},
                  "num_fewshot": 0, "batch_size": 1, "log_samples": True,
                  "bootstrap_iters": 0, "apply_chat_template": task == "gsm8k_cot"}
        if task == "gsm8k_cot":
            kwargs["gen_kwargs"] = {"max_gen_toks": 64, "do_sample": False}
        result = simple_evaluate(**kwargs)
        samples = select_task_samples(task, result["samples"][task])
        if len(samples) != 2:
            raise AssertionError(f"{task} preflight expected two records, got {len(samples)}")
        if PRIMARY[task] not in result["results"][task]:
            raise AssertionError(f"{task} missing official primary metric {PRIMARY[task]}")
        if task in ("hellaswag", "arc_easy"):
            for sample in samples:
                normalized_choice_prediction(task, sample)
        samples_by_task[task] = {"count": len(samples), "metric_keys": list(result["results"][task])}
    rows, _ = load_prompts(config["qualitative_prompts"])
    messages = ([{"role": "system", "content": rows[0]["system"]}] if rows[0]["system"] else [])
    messages.append({"role": "user", "content": rows[0]["prompt"]})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    if output.shape[1] <= inputs["input_ids"].shape[1]:
        raise AssertionError("Qualitative generation produced no tokens")
    samples_by_task["qualitative_smoke"] = {"generated_tokens": output.shape[1]-inputs["input_ids"].shape[1]}
    del model, lm
    gc.collect(); torch.cuda.empty_cache()
    return samples_by_task


def inspect_plotting(config, output_dir):
    from task1.plots import aggregate
    with tempfile.TemporaryDirectory(prefix="plot_smoke_", dir=output_dir) as temp:
        root = Path(temp)
        fixture = {**config, "output_root": str(root / "outputs"), "result_root": str(root / "results")}
        for condition, score in (("baseline", 0.4), ("full_mid", 0.5)):
            path = root / "results" / "evaluation" / condition / "gsm8k_cot.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json({"sample_count": 1319, "results": {"exact_match,flexible-extract": score}}, path)
        path = root / "outputs" / "full" / "train.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"update": 625, "sft_loss": 1.0, "training_seconds": 1.0,
                                    "peak_gpu_bytes": 1, "reference_bytes": 0,
                                    "reference_transfer_seconds": 0}) + "\n", encoding="utf-8")
        summary = aggregate(fixture)
        import pandas as pd
        table = pd.read_csv(summary / "benchmarks.csv")
        change = table.loc[(table.condition == "full_mid") & (table.task == "gsm8k_cot"), "change_pp"].iloc[0]
        if abs(change-10.0) > 1e-9 or not (summary / "benchmarks.png").is_file() or not (summary / "sft_loss.png").is_file():
            raise AssertionError("Plotting fixture did not produce correct scores and figures")
    return {"fixture": "synthetic; never official metrics", "change_pp": change}


def run_preflight(config, requirements_path, cpu_only=False):
    validate_protocol(config)
    out = Path(config["output_root"]) / "preflight"
    out.mkdir(parents=True, exist_ok=True)
    report = {"kind": "Task 1 integration preflight", "cpu_only": cpu_only,
              "checks": {}, "config_sha256": config_sha256(config),
              "source_sha256": source_sha256(), "started_utc": time.time()}

    def check(name, function):
        started = time.perf_counter()
        try:
            detail = function()
            summary = detail if isinstance(detail, (dict, str, int, float, type(None))) else type(detail).__name__
            report["checks"][name] = {"status": "PASSED", "seconds": time.perf_counter()-started,
                                       "detail": summary}
        except Exception as exc:
            report["checks"][name] = {"status": "FAILED", "seconds": time.perf_counter()-started,
                                       "error": f"{type(exc).__name__}: {exc}"}
            if torch.cuda.is_available():
                try:
                    from task1.training import gpu_memory_snapshot
                    report["checks"][name]["gpu_at_failure"] = gpu_memory_snapshot()
                except Exception:
                    pass
            atomic_json(report, out / "report.json", indent=2, default=str)
            raise
        atomic_json(report, out / "report.json", indent=2, default=str)
        return detail

    if cpu_only:
        import subprocess
        import sys
        from task1.qualitative import load_prompts
        check("qualitative_prompt_schema", lambda: {"prompts": len(load_prompts(config["qualitative_prompts"])[0])})
        check("local_unit_suite", lambda: subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
            cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True).stderr.strip())
        check("plotting_serialization_fixture", lambda: inspect_plotting(config, out))
        report["checks"]["gpu_integration"] = {"status": "NOT RUN", "reason": "CPU-only mode"}
        atomic_json(report, out / "report.json", indent=2)
        return report
    (Path(config["output_root"]) / "data" / "preflight_pass.json").unlink(missing_ok=True)
    check("dependencies", lambda: dependency_versions(requirements_path))
    check("resources", lambda: resource_snapshot(config))
    check("plotting_serialization_fixture", lambda: inspect_plotting(config, out))
    tokenizer = check("real_qwen_tokenizer", lambda: inspect_tokenizer(config, out))
    rows, manifest = check("real_prepared_data", lambda: inspect_data(config, tokenizer))
    for method in ("full", "l2sp", "lora4", "lora16", "oplora16"):
        check(f"real_model_optimizer_checkpoint_{method}",
              lambda method=method: inspect_model_method(config, method, rows, tokenizer, out))
    check("real_harness_and_qualitative", lambda: inspect_harness_and_qualitative(config, tokenizer))
    marker = {"data_sha256": manifest["selected_sha256"], "config_sha256": config_sha256(config),
              "source_sha256": source_sha256(),
              "report": str(out / "report.json"), "completed_utc": time.time()}
    atomic_json(marker, Path(config["output_root"]) / "data" / "preflight_pass.json", indent=2)
    return report


def require_passed_preflight(config, manifest):
    path = Path(config["output_root"]) / "data" / "preflight_pass.json"
    if not path.exists():
        raise RuntimeError("Real Kaggle preflight has not passed. Run `python -m task1.run preflight` before training.")
    marker = json.loads(path.read_text(encoding="utf-8"))
    if (marker.get("data_sha256") != manifest["selected_sha256"] or
            marker.get("config_sha256") != config_sha256(config) or
            marker.get("source_sha256") != source_sha256()):
        raise RuntimeError("Preflight marker does not match prepared data, config, and source code")
    report_path = Path(marker["report"])
    if not report_path.is_file():
        raise RuntimeError(f"Preflight report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("cpu_only") or any(check.get("status") != "PASSED" for check in report.get("checks", {}).values()):
        raise RuntimeError("Real Kaggle preflight report is incomplete or failed")
    if len(report.get("checks", {})) < 9:
        raise RuntimeError("Real Kaggle preflight report lacks required checks")
    recorded = report["checks"]["resources"]["detail"]["devices"]
    current = [(torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory)
               for i in range(torch.cuda.device_count())]
    if [(device["name"], device["total_bytes"]) for device in recorded] != current:
        raise RuntimeError("GPU hardware differs from the successful preflight session")
