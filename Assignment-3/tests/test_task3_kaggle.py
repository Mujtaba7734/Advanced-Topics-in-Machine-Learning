"""Exercise the generated Kaggle launcher's persistence without a GPU session."""

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import nbformat
import pytest
from kaggle import task3_dual
from task3.live import CALLS
from task3.storage import Store

NOTEBOOK = Path(__file__).parents[1] / "kaggle/ATML_PA3_Task3_Kaggle_RunAll.ipynb"


@pytest.fixture
def notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def load_function(notebook, name):
    scope = dict(Path=Path, os=os, shutil=shutil, json=json, log=lambda _: None)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        definitions = []
        for statement in ast.parse(cell.source).body:
            if isinstance(statement, ast.FunctionDef) and statement.name != "log":
                definitions.append(statement)
            elif isinstance(statement, ast.Assign) and all(
                isinstance(target, ast.Name) and target.id.isupper() for target in statement.targets
            ) and isinstance(statement.value, ast.Constant):
                definitions.append(statement)
        code = ast.Module(body=definitions, type_ignores=[])
        exec(compile(code, str(NOTEBOOK), "exec"), scope)
        if name in scope:
            return scope[name]
    raise AssertionError(f"Missing notebook function {name}")


def test_dual_gpu_and_original_pipeline(notebook):
    nbformat.validate(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    assert "google.colab" not in code
    assert "b370e572a01f5b0c3cd4d12e0195314be647093f" in code
    assert "CUDA_VISIBLE_DEVICES" in code
    assert "TASK3_T4_DEVICES" in code
    assert "run.run_live = run_live_dual" in code
    assert "run.main()" in code
    assert "/kaggle/working/ATML_PA3_Task3" in code


@pytest.fixture(autouse=True)
def simulated_disk(monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=20_000_000_000))


def test_restore_saved_output_keeps_live_work(notebook, tmp_path):
    restore = load_function(notebook, "restore_saved_output")
    source = tmp_path / "input" / "notebooks" / "owner" / "previous" / "ATML_PA3_Task3"
    target = tmp_path / "working" / source.name
    for root in (source, target):
        (root / "runs" / "experiment").mkdir(parents=True)
    (source / "runs/experiment/old.json").write_text("saved")
    (source / "runs/experiment/shared.json").write_text("old")
    (source / "runs/experiment/interrupted.partial").write_text("interrupted")
    (target / "runs/experiment/shared.json").write_text("current")
    restore(target, tmp_path / "input")
    assert (target / "runs/experiment/old.json").read_text() == "saved"
    assert (target / "runs/experiment/shared.json").read_text() == "current"
    assert not (target / "runs/experiment/interrupted.partial").exists()

    (tmp_path / "input" / "another" / source.name / "runs").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Attach only"):
        restore(target, tmp_path / "input")


def test_model_cache_roundtrip(notebook, tmp_path):
    prepare = load_function(notebook, "prepare_saved_output")
    restore = load_function(notebook, "restore_saved_output")
    source = tmp_path / "input/previous/ATML_PA3_Task3"
    target = tmp_path / "working/ATML_PA3_Task3"
    cache = source / "model_cache"
    model = cache / "models--test--model"
    blob = model / "blobs/abc"
    link = model / "snapshots/revision/weights.safetensors"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"test weights")
    link.parent.mkdir(parents=True)
    link.symlink_to("../../blobs/abc")
    manifest = {
        "model": "test/model",
        "revision": "revision",
        "files": {link.name: hashlib.sha256(blob.read_bytes()).hexdigest()},
    }
    (cache / "revision-verified.json").write_text(json.dumps(manifest))
    (source / "runs").mkdir()
    prepare(source)
    assert not link.is_symlink()
    assert not blob.exists()
    restore(target, tmp_path / "input")
    assert (target / link.relative_to(source)).read_bytes() == b"test weights"


def test_storage_preflight_stops_before_copy(notebook, tmp_path, monkeypatch):
    check = load_function(notebook, "check_storage")
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=7_000_000_000))
    root = tmp_path / "ATML_PA3_Task3"
    root.mkdir()
    with pytest.raises(RuntimeError, match="Insufficient Kaggle space"):
        check(root)


def fake_gpu_worker(slot, device, folder, ids, queue, assigned, checkpoints,
                    identity, cache, logs, deadline, stop):
    """Exercise real spawned processes and checkpoint writes with no CUDA."""
    store = task3_dual.WorkerStore(checkpoints, identity, slot)
    store.put("gpu_smoke", {"simulated": True, "device": device})
    while store.get(f"gpu_smoke_t4_{1 - slot}") is None:
        if time.monotonic() >= deadline:
            raise TimeoutError("Other worker did not start")
        time.sleep(0.01)
    from itertools import chain
    for qid in chain(assigned, iter(queue.get, None)):
        assert store.get(f"live_summary/{qid}") is None
        for name, _ in CALLS:
            key = f"live/{qid}/{name}"
            if store.get(key) is None:
                store.put(key, {"slot": slot})
        store.put(f"live_summary/{qid}", {"slot": slot})
        time.sleep(0.02)


def test_spawned_workers_distribute_questions_and_keep_partial_affinity(tmp_path, monkeypatch):
    store = Store(tmp_path / "checkpoints", "test-identity")
    ids = ["partial", *map(str, range(10))]
    store.put("gpu_affinity/partial", {"slot": 1})
    store.put("live/partial/single", {"slot": 1})
    monkeypatch.setattr(task3_dual, "gpu_worker", fake_gpu_worker)
    monkeypatch.setenv("TASK3_T4_DEVICES", "0,1")
    monitor = SimpleNamespace(deadline=time.monotonic() + 25, log=lambda *args: None)
    task3_dual.run_live_dual(SimpleNamespace(folder=tmp_path), ids, store, monitor, tmp_path)
    assert {store.get(f"live_summary/{qid}")["slot"] for qid in ids} == {0, 1}
    assert {store.get(f"live/partial/{name}")["slot"] for name, _ in CALLS} == {1}
    assert task3_dual.pending_work(ids, store) == ([[], []], [])
