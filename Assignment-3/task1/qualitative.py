import csv
import hashlib
import json
from pathlib import Path
import torch
import yaml
from transformers import AutoTokenizer
from common.checkpoint import atomic_json
from task1.evaluation import CONDITIONS, load_condition, condition_fingerprint


def load_prompts(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Assignment qualitative prompt file missing: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "prompts" in data:
        data = data["prompts"]
    rows = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Each qualitative prompt must be a mapping")
            prompt = item.get("prompt", item.get("text"))
            question_id = str(item.get("question_id", item.get("id")))
            fmt = item.get("format", item.get("type"))
            if prompt is None and "paraphrases" in item:
                paraphrases = item["paraphrases"]
                if not isinstance(paraphrases, list) or len(paraphrases) != 1:
                    raise ValueError(f"Expected one paraphrase for {question_id}")
                prompt = paraphrases[0]
                if question_id.endswith("_json"):
                    question_id, fmt = question_id[:-5], "json"
                elif question_id.endswith("_template"):
                    question_id, fmt = question_id[:-9], "template"
                else:
                    fmt = "plain"
            if prompt is None:
                raise ValueError("Prompt mapping lacks prompt/text")
            rows.append({"question_id": question_id,
                         "format": str(fmt), "prompt": str(prompt),
                         "system": str(item.get("system", ""))})
    elif isinstance(data, dict):
        for question_id, formats in data.items():
            if not isinstance(formats, dict):
                raise ValueError("Expected question ID to format mapping")
            for fmt, prompt in formats.items():
                rows.append({"question_id": str(question_id), "format": str(fmt),
                             "prompt": str(prompt), "system": ""})
    else:
        raise ValueError("Unsupported qualitative YAML structure")
    ids = list(dict.fromkeys(r["question_id"] for r in rows))
    if len(ids) != 8 or len(rows) != 24:
        raise ValueError(f"Expected eight questions and 24 prompts, got {len(ids)} and {len(rows)}")
    for q in ids:
        subset = [r for r in rows if r["question_id"] == q]
        if len(subset) != 3 or len({r["format"].lower() for r in subset}) != 3:
            raise ValueError(f"Question {q} must have three distinct formats")
    return rows, ids


def prompt_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_saved_rows(values, expected_rows, fingerprint):
    if len(values) != len(expected_rows):
        raise ValueError(f"Expected {len(expected_rows)} qualitative records, got {len(values)}")
    for saved, expected in zip(values, expected_rows):
        for key in ("question_id", "format", "prompt", "system"):
            if saved.get(key) != expected[key]:
                raise ValueError(f"Qualitative {key} changed for {expected['question_id']}")
        if saved.get("prompt_sha256") != fingerprint:
            raise ValueError("Qualitative prompt file fingerprint mismatch")


def generate(config, condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    rows, ids = load_prompts(config["qualitative_prompts"])
    fingerprint = prompt_sha256(config["qualitative_prompts"])
    model_fingerprint = condition_fingerprint(config, condition)
    out = Path(config["result_root"]) / "qualitative"
    out.mkdir(parents=True, exist_ok=True)
    atomic_json({"prompt_sha256": fingerprint, "question_ids_plain_order": ids,
                 "prompt_count": len(rows)}, out / "prompts_manifest.json", indent=2)
    file = out / f"{condition}.json"
    outputs = []
    if file.exists():
        outputs = json.loads(file.read_text(encoding="utf-8"))
        if len(outputs) > 24:
            raise ValueError(f"Qualitative output has too many records: {file}")
        for old, expected in zip(outputs, rows):
            for key in ("question_id", "format", "prompt", "system"):
                if (old.get(key) != expected[key] or old.get("prompt_sha256") != fingerprint or
                        old.get("condition_fingerprint") != model_fingerprint):
                    raise ValueError("Qualitative prompt or model changed since partial output was saved")
        if len(outputs) == 24:
            return outputs
    tokenizer = AutoTokenizer.from_pretrained(config["model"], cache_dir=config["cache_dir"],
                                               revision=config.get("model_revision"))
    model = load_condition(config, condition)
    for row in rows[len(outputs):]:
        messages = []
        if row["system"]:
            messages.append({"role": "system", "content": row["system"]})
        messages.append({"role": "user", "content": row["prompt"]})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            sequence = model.generate(**inputs, max_new_tokens=config["qualitative_max_new_tokens"],
                                      do_sample=False, pad_token_id=tokenizer.eos_token_id)
        new = sequence[0, inputs["input_ids"].shape[-1]:]
        outputs.append({**row, "condition": condition, "prompt_sha256": fingerprint,
                        "condition_fingerprint": model_fingerprint,
                        "response": tokenizer.decode(new, skip_special_tokens=True),
                        "generated_tokens": len(new),
                        "truncated": len(new) >= config["qualitative_max_new_tokens"] and
                                     new[-1].item() != tokenizer.eos_token_id})
        atomic_json(outputs, file, indent=2, ensure_ascii=False)
    return outputs


def annotation_template(config):
    rows, ids = load_prompts(config["qualitative_prompts"])
    fingerprint = prompt_sha256(config["qualitative_prompts"])
    out = Path(config["result_root"]) / "qualitative"
    all_outputs = {}
    for condition in CONDITIONS:
        path = out / f"{condition}.json"
        if not path.exists():
            raise FileNotFoundError(f"Qualitative condition missing: {path}")
        values = json.loads(path.read_text(encoding="utf-8"))
        validate_saved_rows(values, rows, fingerprint)
        all_outputs[condition] = {(r["question_id"], r["format"]): r for r in values}
    selected = [r for r in rows if r["question_id"] in (ids[0], ids[-1])]
    fields = ["question_id", "format", "condition", "prompt_sha256", "system", "prompt", "response", "truncated",
              "format_compliance", "coherence_relevance", "repetition",
              "unnecessary_mathematical_framing", "inappropriate_content", "short_excerpt", "notes"]
    with (out / "annotation_template.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in selected:
            for condition in CONDITIONS:
                writer.writerow(all_outputs[condition][(row["question_id"], row["format"])])
    return out / "annotation_template.csv"


def combine_outputs(config):
    out = Path(config["result_root"]) / "qualitative"
    rows, ids = load_prompts(config["qualitative_prompts"])
    fingerprint = prompt_sha256(config["qualitative_prompts"])
    combined = []
    for condition in CONDITIONS:
        path = out / f"{condition}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        values = json.loads(path.read_text(encoding="utf-8"))
        validate_saved_rows(values, rows, fingerprint)
        combined.extend(values)
    combined.sort(key=lambda r: (ids.index(r["question_id"]),
                                  {"plain": 0, "json": 1, "template": 2}.get(r["format"], 3),
                                  list(CONDITIONS).index(r["condition"])))
    fields = ["question_id", "format", "condition", "prompt_sha256", "system", "prompt", "response",
              "generated_tokens", "truncated"]
    with (out / "all_168_responses.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)
    return out / "all_168_responses.csv"
