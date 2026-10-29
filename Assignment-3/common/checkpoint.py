import os
import json
import shutil
from pathlib import Path
import torch


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        torch.save(obj, temp)
        with open(temp, "r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_checked(path, expected):
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"Checkpoint {key} mismatch: {state.get(key)!r} != {value!r}")
    return state


def atomic_json(obj, path, **json_kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(obj, handle, **json_kwargs)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_latest(source, target):
    """Point latest at a complete recovery file without a second large write."""
    source, target = Path(source), Path(target)
    temp = target.with_name(target.name + ".tmp")
    try:
        try:
            os.link(source, temp)
        except OSError:
            shutil.copyfile(source, temp)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
