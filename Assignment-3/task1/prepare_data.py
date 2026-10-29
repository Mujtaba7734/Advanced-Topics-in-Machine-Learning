"""Deterministic MetaMathQA selection. No training examples are shuffled."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from common.checkpoint import atomic_bytes, atomic_json


def normalize_question(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(re.findall(r"\w+", text))


def token_ids(value):
    """Unwrap a tokenizer BatchEncoding or a flat token list without guessing shape."""
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(x, int) for x in value):
        raise TypeError(f"Expected one flat token-ID list, received {type(value).__name__}")
    return value


def encode_example(tokenizer, question, response):
    if not question.strip() or not response.strip():
        raise ValueError("Question and response must be nonempty")
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    if not isinstance(prompt_text, str) or not prompt_text:
        raise TypeError("Qwen chat template did not return a prompt string")
    end_token = "<|im_end|>"
    end_id = tokenizer.convert_tokens_to_ids(end_token)
    if not isinstance(end_id, int) or end_id == tokenizer.unk_token_id:
        raise ValueError("Qwen end-of-turn token is unavailable")
    prompt = token_ids(tokenizer(prompt_text, add_special_tokens=False))
    answer = token_ids(tokenizer(response, add_special_tokens=False))
    if not answer or end_id in answer:
        raise ValueError("Empty response tokens or embedded end-of-turn token")
    complete = prompt + answer + [end_id]
    expected_text = prompt_text + response + end_token
    decoded = tokenizer.decode(complete, skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
    if decoded != expected_text:
        raise ValueError("Qwen tokenization did not round-trip the formatted example")
    return {"input_ids": complete, "labels": [-100] * len(prompt) + answer + [end_id],
            "prompt_tokens": len(prompt), "response_tokens": len(answer) + 1}


def select_examples(rows, gsm_questions, tokenizer, count=40000, max_length=2048):
    contamination = {normalize_question(q) for q in gsm_questions}
    selected, rejected = [], {"empty": 0, "contaminated": 0, "too_long": 0}
    for source_index, row in enumerate(rows):
        question = str(row.get("query", "") or "")
        answer = str(row.get("response", "") or "")
        if not question.strip() or not answer.strip():
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
    if (out / "selected.json").exists() and (out / "manifest.json").exists():
        from task1.dataset import load_selected
        _, previous = load_selected(config)
        if previous.get("model") != config["model"]:
            raise ValueError("Prepared examples belong to a different model")
        print("Validated existing 40,000-example selection; preparation is complete", flush=True)
        return previous
    tokenizer = AutoTokenizer.from_pretrained(config["model"], cache_dir=config["cache_dir"],
                                               revision=config.get("model_revision"))
    model_config = AutoConfig.from_pretrained(config["model"], cache_dir=config["cache_dir"],
                                               revision=config.get("model_revision"))
    gsm = load_split(config["gsm8k_dataset"], "test", config["cache_dir"], "main")
    if len(gsm) != 1319:
        raise RuntimeError(f"GSM8K test count changed: {len(gsm)}")
    if "question" not in gsm.column_names:
        raise ValueError(f"GSM8K test schema lacks question: {gsm.column_names}")
    source = load_split(config["metamath_dataset"], "train", config["cache_dir"])
    required = {"query", "response", "original_question"}
    if not required.issubset(source.column_names):
        raise ValueError(f"MetaMathQA schema missing {sorted(required-set(source.column_names))}")
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
    atomic_bytes(payload, out / "selected.json")
    atomic_json(manifest, out / "manifest.json", indent=2)
    return manifest
