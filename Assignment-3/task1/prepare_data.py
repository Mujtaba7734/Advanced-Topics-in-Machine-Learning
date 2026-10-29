"""Deterministic MetaMathQA selection. No training examples are shuffled."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def normalize_question(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(re.findall(r"\w+", text))


def encode_example(tokenizer, question, response):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    complete = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}, {"role": "assistant", "content": response}],
        tokenize=True, add_generation_prompt=False, enable_thinking=False)
    if complete[:len(prompt)] != prompt:
        raise ValueError("Chat template assistant prefix does not match prompt prefix")
    if len(complete) == len(prompt):
        raise ValueError("Empty assistant target")
    return {"input_ids": complete, "labels": [-100] * len(prompt) + complete[len(prompt):],
            "response_tokens": len(complete) - len(prompt)}


def select_examples(rows, gsm_questions, tokenizer, count=40000, max_length=2048):
    contamination = {normalize_question(q) for q in gsm_questions}
    selected, rejected = [], {"empty": 0, "contaminated": 0, "too_long": 0}
    for source_index, row in enumerate(rows):
        question = str(row.get("query", "") or "").strip()
        answer = str(row.get("response", "") or "").strip()
        if not question or not answer:
            rejected["empty"] += 1
            continue
        original = row.get("original_question", "") or ""
        if normalize_question(question) in contamination or normalize_question(original) in contamination:
            rejected["contaminated"] += 1
            continue
        encoded = encode_example(tokenizer, question, answer)
        if len(encoded["input_ids"]) > max_length:
            rejected["too_long"] += 1
            continue
        encoded["source_index"] = source_index
        encoded["source_id"] = str(row.get("id", source_index))
        selected.append(encoded)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Only {len(selected)} eligible examples; required {count}")
    return selected, rejected


def prepare(config):
    from transformers import AutoTokenizer, AutoConfig
    from task1.data_source import load_split
    out = Path(config["output_root"]) / "data"
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], cache_dir=config["cache_dir"],
                                               revision=config.get("model_revision"))
    model_config = AutoConfig.from_pretrained(config["model"], cache_dir=config["cache_dir"],
                                               revision=config.get("model_revision"))
    gsm = load_split(config["gsm8k_dataset"], "test", config["cache_dir"], "main")
    if len(gsm) != 1319:
        raise RuntimeError(f"GSM8K test count changed: {len(gsm)}")
    source = load_split(config["metamath_dataset"], "train", config["cache_dir"])
    selected, rejected = select_examples(source, (r["question"] for r in gsm), tokenizer,
                                          config["train_examples"], config["max_length"])
    manifest = {"source_dataset": config["metamath_dataset"], "source_fingerprint": source._fingerprint,
                "gsm8k_fingerprint": gsm._fingerprint, "model": config["model"],
                "model_revision": getattr(model_config, "_commit_hash", None) or
                                  tokenizer.init_kwargs.get("_commit_hash") or config.get("model_revision"),
                "tokenizer_revision": tokenizer.init_kwargs.get("revision"),
                "indices": [r["source_index"] for r in selected],
                "ids": [r["source_id"] for r in selected], "rejected": rejected,
                "length_min": min(len(r["input_ids"]) for r in selected),
                "length_max": max(len(r["input_ids"]) for r in selected),
                "length_mean": sum(len(r["input_ids"]) for r in selected) / len(selected),
                "response_tokens": sum(r["response_tokens"] for r in selected),
                "response_tokens_min": min(r["response_tokens"] for r in selected),
                "response_tokens_max": max(r["response_tokens"] for r in selected),
                "source_rows_examined": selected[-1]["source_index"] + 1,
                "normalization": "NFKC casefold, Unicode word tokens joined by single spaces"}
    payload = json.dumps(selected, separators=(",", ":")).encode()
    manifest["selected_sha256"] = hashlib.sha256(payload).hexdigest()
    (out / "selected.json").write_bytes(payload)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
