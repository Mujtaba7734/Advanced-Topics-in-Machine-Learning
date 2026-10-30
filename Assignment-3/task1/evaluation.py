"""Full-split zero-shot evaluation using pinned lm-evaluation-harness tasks."""
import json
import hashlib
from pathlib import Path
import torch
from transformers import AutoTokenizer
from common.checkpoint import atomic_json
from task1.model import load_model, load_adapter_state, merge_adapters, use_projected_factors
from task1.protocol import config_sha256


CONDITIONS = {
    "baseline": (None, None), "full_mid": ("full", 625), "full_final": ("full", 1250),
    "l2sp_final": ("l2sp", 1250), "lora4_final": ("lora4", 1250),
    "lora16_final": ("lora16", 1250), "oplora16_final": ("oplora16", 1250),
}
TASKS = {"gsm8k_cot": 1319, "hellaswag": 10042, "boolq": 3270, "arc_easy": 2376}

# Preserve task protocol while maximizing inference throughput on a 16-GiB T4.
# lm-eval probes from 32 downwards on OOM and recalibrates across context lengths.
EVAL_BATCH_SIZE = "auto:4"
EVAL_MAX_BATCH_SIZE = 32


def condition_fingerprint(config, condition):
    method, step = CONDITIONS[condition]
    digest = hashlib.sha256()
    digest.update(config_sha256(config).encode())
    digest.update(condition.encode())
    if method:
        path = Path(config["output_root"]) / method / f"inference_{step:04d}.pt"
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    else:
        local_model = Path(config["model"])
        if local_model.is_dir():
            weights = sorted(local_model.glob("*.safetensors")) + sorted(local_model.glob("*.bin"))
            if not weights:
                raise FileNotFoundError(f"Local baseline model has no weight files: {local_model}")
            for weight in weights:
                digest.update(weight.name.encode())
                with weight.open("rb") as handle:
                    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(block)
        else:
            digest.update(str(config.get("model_revision") or config["model"]).encode())
    return digest.hexdigest()


def answer_from_sample(sample):
    filtered = sample.get("filtered_resps", [])
    if isinstance(filtered, list) and filtered:
        return filtered[0]
    return None


def select_task_samples(task, samples):
    if task == "gsm8k_cot":
        return [sample for sample in samples if sample.get("filter") == "flexible-extract"]
    return samples


def multiple_choice_prediction(sample):
    responses = sample.get("filtered_resps", [])
    if not responses:
        return None
    scores = [float(response[0] if isinstance(response, (list, tuple)) else response)
              for response in responses]
    return max(range(len(scores)), key=scores.__getitem__)


def normalized_choice_prediction(task, sample):
    if task not in ("hellaswag", "arc_easy"):
        raise ValueError(f"No normalized-choice metric for {task}")
    choices = sample["doc"]["choices"]
    if task == "arc_easy":
        choices = choices["text"]
    responses = sample.get("filtered_resps", [])
    if len(choices) != len(responses) or any(len(choice) == 0 for choice in choices):
        raise ValueError(f"{task} choices/responses mismatch for normalized prediction")
    scores = [float(response[0] if isinstance(response, (list, tuple)) else response) / len(choice)
              for response, choice in zip(responses, choices)]
    return max(range(len(scores)), key=scores.__getitem__)


def load_condition(config, condition):
    method, step = CONDITIONS[condition]
    actual_method = method or "full"
    model, _ = load_model(config, actual_method, for_inference=True)
    if method:
        path = Path(config["output_root"]) / method / f"inference_{step:04d}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Required inference checkpoint missing: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if method in ("full", "l2sp"):
            model.load_state_dict(state)
        else:
            load_adapter_state(model, state)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(config["model"], cache_dir=config["cache_dir"],
                                                       revision=config.get("model_revision"))
            ids = tokenizer("A short merge equivalence check.", return_tensors="pt").input_ids.cuda()
            with torch.inference_mode():
                before = model(input_ids=ids).logits.float()
            if method == "oplora16":
                projected = torch.load(path.with_name(f"projected_{step:04d}.pt"),
                                       map_location="cpu", weights_only=True)
                use_projected_factors(model, projected)
                with torch.inference_mode():
                    projected_logits = model(input_ids=ids).logits.float()
                projected_error = (before - projected_logits).abs().max().item()
                if projected_error > 0.1:
                    raise AssertionError(f"Projected export differs: {projected_error}")
            merge_adapters(model)
            with torch.inference_mode():
                after = model(input_ids=ids).logits.float()
            error = (before - after).abs().max().item()
            if error > 0.1:
                raise AssertionError(f"Merged logits differ: {error}")
    model.eval()
    model.config.use_cache = True
    return model


def evaluate(config, condition):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    if condition not in CONDITIONS:
        raise ValueError(condition)
    out = Path(config["result_root"]) / "evaluation" / condition
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = condition_fingerprint(config, condition)
    aggregates = {}
    pending = []
    for task, expected in TASKS.items():
        target = out / f"{task}.json"
        if target.exists():
            saved = json.loads(target.read_text(encoding="utf-8"))
            if saved.get("condition_fingerprint") != fingerprint:
                raise ValueError(f"Stale evaluation result has a different model/config fingerprint: {target}")
            if saved.get("sample_count") == expected:
                aggregates[task] = saved["results"]
                continue
        pending.append((task, expected, target))
    if pending:
        tokenizer = AutoTokenizer.from_pretrained(config["model"], cache_dir=config["cache_dir"],
                                                   revision=config.get("model_revision"))
        model = load_condition(config, condition)
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=EVAL_BATCH_SIZE,
                  max_batch_size=EVAL_MAX_BATCH_SIZE, enable_thinking=False)
    for task, expected, target in pending:
        print(f"[EVAL] {condition}/{task}: auto batch up to {EVAL_MAX_BATCH_SIZE}", flush=True)
        kwargs = {"model": lm, "tasks": [task], "num_fewshot": 0,
                  "batch_size": EVAL_BATCH_SIZE, "max_batch_size": EVAL_MAX_BATCH_SIZE,
                  "log_samples": True, "limit": None, "bootstrap_iters": 0,
                  "apply_chat_template": task == "gsm8k_cot",
                  "random_seed": config["seed"], "numpy_random_seed": config["seed"],
                  "torch_random_seed": config["seed"], "fewshot_random_seed": config["seed"]}
        if task == "gsm8k_cot":
            kwargs["gen_kwargs"] = {"max_gen_toks": config["gsm8k_max_new_tokens"], "do_sample": False}
        result = simple_evaluate(**kwargs)
        samples = select_task_samples(task, result["samples"][task])
        if len(samples) != expected:
            raise RuntimeError(f"{task} returned {len(samples)} samples, expected {expected}")
        for sample in samples:
            sample["prediction_raw"] = (None if task == "gsm8k_cot" else multiple_choice_prediction(sample))
            sample["prediction"] = (answer_from_sample(sample) if task == "gsm8k_cot" else
                                    normalized_choice_prediction(task, sample) if task in ("hellaswag", "arc_easy") else
                                    sample["prediction_raw"])
            sample["extracted_answer"] = answer_from_sample(sample) if task == "gsm8k_cot" else None
            if task == "gsm8k_cot":
                raw = sample.get("resps", [])
                text = raw[0][0] if raw and isinstance(raw[0], (list, tuple)) else (raw[0] if raw else "")
                sample["truncation_flag"] = len(tokenizer.encode(str(text), add_special_tokens=False)) >= config["gsm8k_max_new_tokens"]
                sample["truncation_flag_kind"] = "retokenized_text_lower_bound"
            else:
                sample["truncation_flag"] = False
        payload = {"condition": condition, "task": task, "sample_count": len(samples),
                   "condition_fingerprint": fingerprint,
                   "results": result["results"][task], "samples": samples,
                   "config": result.get("config", {})}
        atomic_json(payload, target, indent=2, default=str)
        aggregates[task] = payload["results"]
    atomic_json(aggregates, out / "aggregate.json", indent=2)
    return aggregates
