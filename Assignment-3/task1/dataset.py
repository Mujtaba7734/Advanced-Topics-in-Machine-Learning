import json
import hashlib
from pathlib import Path
import torch


def load_selected(config):
    root = Path(config["output_root"]) / "data"
    payload = (root / "selected.json").read_bytes()
    rows = json.loads(payload)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(payload).hexdigest() != manifest["selected_sha256"]:
        raise ValueError("Selected dataset SHA256 mismatch")
    if len(rows) != config["train_examples"]:
        raise ValueError("Selected dataset count mismatch")
    if [r["source_index"] for r in rows] != manifest["indices"]:
        raise ValueError("Selected dataset ordering mismatch")
    return rows, manifest


def collate(rows, pad_token_id, device):
    width = max(len(row["input_ids"]) for row in rows)
    ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        n = len(row["input_ids"])
        ids[i, :n] = torch.tensor(row["input_ids"])
        labels[i, :n] = torch.tensor(row["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids.to(device), "labels": labels.to(device), "attention_mask": mask.to(device)}
