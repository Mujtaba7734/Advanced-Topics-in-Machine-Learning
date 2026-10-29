"""Weight and common-baseline activation diagnostics for fixed layers."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from task1.model import load_model, leading_bases
from task1.evaluation import CONDITIONS
from task1.protocol import config_sha256
from common.checkpoint import atomic_save, atomic_json


def metrics(weight, delta, uk):
    w, d, u = weight.float(), delta.float(), uk.float()
    dnorm = torch.linalg.vector_norm(d)
    rho = 0.0 if dnorm == 0 else ((u.T @ d).square().sum() / dnorm.square()).item()
    eta = (dnorm / torch.linalg.vector_norm(w)).item()
    return {"rho": rho, "eta": eta, "zero_update": bool(dnorm == 0)}


def activation_ratio(weight, delta, z):
    return (torch.linalg.vector_norm(z.float() @ delta.float().T, dim=-1) /
            (torch.linalg.vector_norm(z.float() @ weight.float().T, dim=-1) + 1e-8))


def layer_key(i):
    return f"model.layers.{i}.mlp.down_proj"


def correction(config, condition, layer, original):
    method, step = CONDITIONS[condition]
    path = Path(config["output_root"]) / method / f"inference_{step:04d}.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    return correction_from_state(config, condition, layer, original, state)


def correction_from_state(config, condition, layer, original, state):
    method, _ = CONDITIONS[condition]
    key = layer_key(layer)
    if method in ("full", "l2sp"):
        return state[key + ".weight"].float() - original.float()
    a, b = state[key + ".A"].float(), state[key + ".B"].float()
    if method == "oplora16":
        uk, vk = state[key + ".uk"].float(), state[key + ".vk"].float()
        b = b - uk @ (uk.T @ b)
        a = a - (a @ vk) @ vk.T
    rank = 4 if method == "lora4" else 16
    alpha = 8 if method == "lora4" else 32
    return (alpha / rank) * (b @ a)


def original_layers(config):
    model, _ = load_model(config, "full", for_inference=False)
    result = {i: model.model.layers[i].mlp.down_proj.weight.detach().cpu().float().clone()
              for i in config["diagnostic_layers"]}
    del model
    torch.cuda.empty_cache()
    return result


def run_weight_diagnostics(config):
    out = Path(config["result_root"]) / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    weights = original_layers(config)
    rows = []
    bases = {layer: leading_bases(weight, config["op_rank"])[0] for layer, weight in weights.items()}
    for condition, (method, step) in CONDITIONS.items():
        if condition == "baseline":
            continue
        state = torch.load(Path(config["output_root"]) / method / f"inference_{step:04d}.pt",
                           map_location="cpu", weights_only=True)
        for layer, weight in weights.items():
            delta = correction_from_state(config, condition, layer, weight, state)
            rows.append({"condition": condition, "layer": layer, **metrics(weight, delta, bases[layer])})
    pd.DataFrame(rows).to_csv(out / "weight_space.csv", index=False)
    return rows


def create_probe_ids(config):
    from task1.data_source import load_split
    out = Path(config["output_root"]) / "data" / "probe_ids.json"
    if out.exists():
        ids = json.loads(out.read_text(encoding="utf-8"))
        for label, upper in (("gsm8k", 1319), ("hellaswag", 10042)):
            values = ids.get(label, [])
            if len(values) != 32 or values != sorted(set(values)) or any(
                    not isinstance(value, int) or value < 0 or value >= upper for value in values):
                raise ValueError(f"Invalid saved {label} probe IDs")
        return ids
    gsm = load_split(config["gsm8k_dataset"], "test", config["cache_dir"], "main")
    hs = load_split(config["hellaswag_dataset"], "validation", config["cache_dir"])
    if len(gsm) != 1319 or len(hs) != 10042:
        raise RuntimeError("Probe source split sizes changed")
    rng = np.random.default_rng(config["seed"])
    ids = {"gsm8k": sorted(rng.choice(len(gsm), 32, replace=False).tolist()),
           "hellaswag": sorted(rng.choice(len(hs), 32, replace=False).tolist())}
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(ids, out, indent=2)
    return ids


def cache_activations(config):
    from lm_eval.tasks import get_task_dict
    out = Path(config["output_root"]) / "data" / "baseline_activations.pt"
    if out.exists():
        cache = torch.load(out, map_location="cpu", weights_only=True)
        if cache.get("config_sha256") != config_sha256(config) or cache.get("ids") != create_probe_ids(config):
            raise ValueError("Baseline activation cache is stale for config/probe selection")
        return out
    ids = create_probe_ids(config)
    tasks = get_task_dict(["gsm8k_cot", "hellaswag"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"], cache_dir=config["cache_dir"],
                                               revision=config.get("model_revision"))
    model, _ = load_model(config, "full", for_inference=True)
    model.eval()
    storage = {i: {"gsm8k": [], "hellaswag": []} for i in config["diagnostic_layers"]}
    captured = {}
    handles = []
    for i in config["diagnostic_layers"]:
        handles.append(model.model.layers[i].mlp.down_proj.register_forward_hook(
            lambda module, args, result, layer=i: captured.__setitem__(layer, args[0][:, -1, :].detach().cpu().float())))
    try:
        for label, task_name in (("gsm8k", "gsm8k_cot"), ("hellaswag", "hellaswag")):
            task = tasks[task_name]
            docs = list(task.test_docs() if label == "gsm8k" else task.validation_docs())
            for index in ids[label]:
                text = task.doc_to_text(docs[index])
                if label == "gsm8k":
                    text = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
                tokens = tokenizer(text, return_tensors="pt").to("cuda")
                captured.clear()
                with torch.inference_mode():
                    model(**tokens)
                for i in config["diagnostic_layers"]:
                    storage[i][label].append(captured[i].squeeze(0))
    finally:
        for h in handles:
            h.remove()
    data = {i: {label: torch.stack(values) for label, values in groups.items()}
            for i, groups in storage.items()}
    atomic_save({"ids": ids, "activations": data, "config_sha256": config_sha256(config)}, out)
    return out


def run_activation_diagnostics(config):
    cache = torch.load(cache_activations(config), map_location="cpu", weights_only=False)
    weights = original_layers(config)
    rows = []
    for condition, (method, step) in CONDITIONS.items():
        if condition == "baseline":
            continue
        state = torch.load(Path(config["output_root"]) / method / f"inference_{step:04d}.pt",
                           map_location="cpu", weights_only=True)
        for layer, weight in weights.items():
            delta = correction_from_state(config, condition, layer, weight, state)
            for group in ("gsm8k", "hellaswag"):
                values = activation_ratio(weight, delta, cache["activations"][layer][group])
                for source_id, value in zip(cache["ids"][group], values.tolist()):
                    rows.append({"condition": condition, "layer": layer, "group": group,
                                 "source_id": source_id, "ratio": value})
    out = Path(config["result_root"]) / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "activation_ratios.csv", index=False)
    return rows
