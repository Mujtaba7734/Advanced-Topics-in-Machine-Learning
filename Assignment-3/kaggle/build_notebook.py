"""Generate the audited source manifest and 18-stage Kaggle Run All notebook."""
import hashlib
import inspect
import json
import os
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    ("Environment and repository setup", "environment"),
    ("Dependency installation and verification", "dependencies"),
    ("Previous outputs and recovery", "restore"),
    ("Dataset preparation", "dataset"),
    ("GPU preflight", "preflight"),
    ("Baseline evaluation", "baseline"),
    ("Full fine-tuning", "train_full"),
    ("L2-SP training", "train_l2sp"),
    ("LoRA rank 4 training", "train_lora4"),
    ("LoRA rank 16 training", "train_lora16"),
    ("OPLoRA rank 16 training", "train_oplora16"),
    ("Six trained-model evaluations", "evaluate_models"),
    ("Weight-space diagnostics", "weight"),
    ("Activation-space diagnostics", "activations"),
    ("Qualitative generation", "qualitative"),
    ("Results aggregation", "aggregate"),
    ("Final verification", "final_verify"),
    ("Export and backup manifest", "export"),
]


def digest(path):
    content = Path(path).read_bytes()
    # Git stores audited text with LF even if a Windows worktree has CRLF.
    if Path(path).suffix.lower() != ".pdf":
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def write_manifest():
    paths = (list((ROOT / "task1").glob("*.py")) + list((ROOT / "common").glob("*.py")) +
             [ROOT / "configs/task1.yaml", ROOT / "requirements.txt",
              ROOT / "first_plot_questions.yaml", ROOT / "ATML-PA3.pdf",
              ROOT / "docs/task1_traceability.md", ROOT / ".gitattributes",
              ROOT / "kaggle/__init__.py", ROOT / "kaggle/runall.py",
              ROOT / "kaggle/build_notebook.py"])
    files = {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(paths)}
    revision = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {"schema": 1, "revision": revision, "files": files}
    target = ROOT / "kaggle" / "source_manifest.json"
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    recorded = json.loads(target.read_text(encoding="utf-8"))
    for relative, expected in recorded["files"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"Source changed during manifest generation: {relative}")
    if recorded["revision"] != hashlib.sha256(json.dumps(
            recorded["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest():
        raise RuntimeError("Source manifest revision is inconsistent")
    return digest(target), revision


def source_matches(path, expected_manifest_sha256):
    """Verify the manifest bytes and every recorded source file."""
    try:
        manifest_path = path / "kaggle/source_manifest.json"
        payload = manifest_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_manifest_sha256:
            return False
        manifest = json.loads(payload)
        calculated = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()
        return (manifest["revision"] == calculated and
                all((path / relative).is_file() and
                    hashlib.sha256((path / relative).read_bytes()).hexdigest() == expected
                    for relative, expected in manifest["files"].items()))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def repository_candidates(working, input_root):
    candidates = [working / "Assignment-3", working / "atml-repo" / "Assignment-3",
                  working / "atml-repo-audited" / "Assignment-3"]
    candidates += sorted(working.glob("atml-repo-*/Assignment-3"))
    if input_root.is_dir():
        candidates += sorted(input_root.glob("*/Assignment-3"))
        candidates += sorted(input_root.glob("*"))
    return candidates


def next_clone_path(working, expected_manifest_sha256):
    """Choose a new content-addressed checkout without changing a stale clone."""
    base = working / ("atml-repo-" + expected_manifest_sha256[:16])
    clone = base
    repair = 0
    while clone.exists():
        repair += 1
        clone = working / (base.name + "-repair-" + str(repair))
    return clone


def stage_cell(stage, manifest_hash):
    helpers = "\n\n".join(inspect.getsource(func) for func in
                         (source_matches, repository_candidates, next_clone_path))
    return f'''# This cell is self-contained; it also works after a kernel restart.
from pathlib import Path
import hashlib, json, subprocess, sys

EXPECTED_MANIFEST_SHA256 = "{manifest_hash}"
working = Path("/kaggle/working")
input_root = Path("/kaggle/input")
{helpers}

matching = [path for path in repository_candidates(working, input_root)
            if source_matches(path, EXPECTED_MANIFEST_SHA256)]
if matching:
    repo = matching[0]
else:
    # A fresh checkout has a content-addressed path; stale clones remain untouched.
    clone = next_clone_path(working, EXPECTED_MANIFEST_SHA256)
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                    "https://github.com/Mujtaba7734/Advanced-Topics-in-Machine-Learning.git",
                    str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "sparse-checkout", "set", "Assignment-3"], check=True)
    repo = clone / "Assignment-3"
    if not source_matches(repo, EXPECTED_MANIFEST_SHA256):
        raise RuntimeError("The public checkout does not contain this notebook's exact audited source. Attach the matching Assignment-3 folder as a Kaggle input; existing outputs were preserved.")
print("[{stage}] source:", repo, flush=True)
subprocess.run([sys.executable, "-m", "kaggle.runall", "{stage}",
                "--repo", str(repo), "--work", str(working)], cwd=repo, check=True)
'''


def main():
    manifest_hash, revision = write_manifest()
    notebook = nbf.v4.new_notebook()
    intro = nbf.v4.new_markdown_cell(
        "# ATML PA3 Task 1 — Run All / Resume All\n\n"
        "Select T4 ×2 and enable Internet, attach the current audited Assignment-3 source "
        "and any previously saved Task 1 output dataset, then click **Run All**. "
        "Each stage is a separate, self-contained code cell. A staged training stop "
        "requires Kaggle **Save Version** and mounting that output dataset before the next Run All. "
        "No cell changes the assignment protocol. The written report remains the student's work.")
    intro["id"] = "task1-intro"
    notebook["cells"] = [intro]
    for index, (title, stage) in enumerate(STAGES, 1):
        heading = nbf.v4.new_markdown_cell(f"## {index}. {title}")
        heading["id"] = f"heading-{index:02d}"
        cell = nbf.v4.new_code_cell(stage_cell(stage, manifest_hash))
        cell["id"] = f"stage-{index:02d}"
        notebook["cells"].extend((heading, cell))
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "task1_source_revision": revision,
        "task1_manifest_sha256": manifest_hash,
    }
    target = ROOT / "kaggle" / "ATML_PA3_Task1_RunAll.ipynb"
    nbf.write(notebook, target)
    print(target)
    print(f"Source revision: {revision}")


if __name__ == "__main__":
    main()
