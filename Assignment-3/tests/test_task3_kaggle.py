"""Exercise the generated Kaggle launcher's persistence without a GPU session."""

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil

import nbformat
import pytest

NOTEBOOK = Path(__file__).parents[1] / "kaggle/ATML_PA3_Task3_Kaggle_RunAll.ipynb"


@pytest.fixture
def notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def load_function(notebook, name):
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for statement in ast.parse(cell.source).body:
            if isinstance(statement, ast.FunctionDef) and statement.name == name:
                scope = dict(Path=Path, os=os, shutil=shutil, json=json, log=lambda _: None)
                code = ast.Module(body=[statement], type_ignores=[])
                exec(compile(code, str(NOTEBOOK), "exec"), scope)
                return scope[name]
    raise AssertionError(f"Missing notebook function {name}")


def test_single_gpu_and_original_pipeline(notebook):
    nbformat.validate(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    assert "google.colab" not in code
    assert "b370e572a01f5b0c3cd4d12e0195314be647093f" in code
    assert "CUDA_VISIBLE_DEVICES" in code
    assert "'-m', 'task3.run'" in code
    assert "/kaggle/working/ATML_PA3_Task3" in code


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
