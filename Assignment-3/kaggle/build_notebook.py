"""Build the self-contained Kaggle phase controller."""
from pathlib import Path
import hashlib
import nbformat as nbf


controller = '''from pathlib import Path
import hashlib, shutil, subprocess, sys

# Edit these controls; run this cell once per phase and session.
PHASE = "diagnose"  # diagnose, prepare, preflight, train, evaluate, weight, activations, qualitative, aggregate, export
METHOD = "full"  # full, l2sp, lora4, lora16, oplora16
CONDITION = "baseline"  # baseline, full_mid, full_final, l2sp_final, lora4_final, lora16_final, oplora16_final
STOP_UPDATE = None  # 312, 625, 937, 1250, or None
INSTALL_REQUIREMENTS = True
REPO_SOURCE = None  # optional mounted Assignment-3 root
RESTORE_DATASETS = []  # mounted previous output datasets

working = Path("/kaggle/working")
def valid_repo(path):
    return (path / "task1" / "run.py").is_file() and (path / "requirements.txt").is_file()

if REPO_SOURCE:
    repo = Path(REPO_SOURCE)
    if not valid_repo(repo):
        raise FileNotFoundError(f"Invalid Assignment-3 root: {repo}")
else:
    candidates = [working / "atml-repo" / "Assignment-3", working / "Assignment-3"]
    candidates += list(Path("/kaggle/input").glob("*/Assignment-3"))
    candidates += list(Path("/kaggle/input").glob("*"))
    repo = next((p for p in candidates if valid_repo(p)), None)
    if repo is None:
        clone = working / "atml-repo"
        if clone.exists():
            raise FileNotFoundError(f"Existing clone lacks Assignment-3/task1/run.py: {clone}")
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/Mujtaba7734/Advanced-Topics-in-Machine-Learning.git",
                        str(clone)], check=True)
        subprocess.run(["git", "-C", str(clone), "sparse-checkout", "set", "Assignment-3"], check=True)
        repo = clone / "Assignment-3"
        if not valid_repo(repo):
            raise FileNotFoundError(f"Clone lacks Assignment-3/task1/run.py: {repo}")
print("Repository:", repo, flush=True)

def copy_verified(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        if digest(source) != digest(target):
            raise ValueError(f"Conflicting restored file: {target}. Use one current output dataset.")
        return
    shutil.copy2(source, target)

for mount in map(Path, RESTORE_DATASETS):
    if not mount.is_dir():
        raise FileNotFoundError(mount)
    for relative in ("outputs/task1/data", "outputs/task1/preflight", "results/task1"):
        source_dir = mount / relative
        if source_dir.exists():
            for source in source_dir.rglob("*"):
                if source.is_file():
                    copy_verified(source, working / relative / source.relative_to(source_dir))
    methods = [METHOD] if PHASE in ("train", "export") else ["full", "l2sp", "lora4", "lora16", "oplora16"]
    for method in methods:
        source_method = mount / "outputs/task1" / method
        if not source_method.exists():
            continue
        recovery = sorted(source_method.glob("recovery_*.pt"))
        for source in source_method.iterdir():
            if not source.is_file() or (source.name.startswith("recovery_") and PHASE != "export"):
                continue
            if PHASE not in ("train", "export") and source.name == "latest.pt":
                continue
            copy_verified(source, working / "outputs/task1" / method / source.name)
        if PHASE == "train" and not (working / "outputs/task1" / method / "latest.pt").exists() and recovery:
            copy_verified(recovery[-1], working / "outputs/task1" / method / recovery[-1].name)
    print("Restored:", mount, flush=True)

if INSTALL_REQUIREMENTS:
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(repo / "requirements.txt")], check=True)

import yaml
config = yaml.safe_load((repo / "configs/task1.yaml").read_text(encoding="utf-8"))
prompt_path = repo / "first_plot_questions.yaml"
if not prompt_path.is_file():
    matches = list(Path("/kaggle/input").glob("*/first_plot_questions.yaml"))
    matches += list(Path("/kaggle/input").glob("*/Assignment-3/first_plot_questions.yaml"))
    if len(matches) != 1:
        raise FileNotFoundError("Mount first_plot_questions.yaml, or include it in the repository")
    prompt_path = matches[0]
config["qualitative_prompts"] = str(prompt_path)
runtime_config = working / "task1_runtime.yaml"
runtime_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print("Prompt file:", prompt_path, flush=True)

cmd = [sys.executable, "-m", "task1.run", PHASE, "--config", str(runtime_config)]
if PHASE in ("train", "export"):
    cmd += ["--method", METHOD]
if PHASE == "train" and STOP_UPDATE is not None:
    cmd += ["--stop-update", str(STOP_UPDATE)]
if PHASE in ("evaluate", "qualitative"):
    cmd += ["--condition", CONDITION]
print("Running:", " ".join(cmd), flush=True)
subprocess.run(cmd, cwd=repo, check=True)
'''

source_root = Path(__file__).resolve().parents[1]
audited_paths = (list((source_root / "task1").glob("*.py")) +
                 list((source_root / "common").glob("*.py")) +
                 [source_root / "configs/task1.yaml", source_root / "requirements.txt",
                  source_root / "kaggle/build_notebook.py"])
expected_hashes = {str(path.relative_to(source_root)).replace("\\", "/"):
                   hashlib.sha256(path.read_bytes()).hexdigest() for path in audited_paths}
guard = ("EXPECTED_SOURCE_HASHES = " + repr(expected_hashes) + "\n"
         "for relative, expected in EXPECTED_SOURCE_HASHES.items():\n"
         "    source = repo / relative\n"
         "    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:\n"
         "        raise RuntimeError(f'Audited code mismatch at {source}. Upload the current Assignment-3 folder as a Kaggle input.')\n")
controller = controller.replace('print("Repository:", repo, flush=True)\n',
                                'print("Repository:", repo, flush=True)\n' + guard, 1)

nb = nbf.v4.new_notebook()
nb["cells"] = [
    nbf.v4.new_markdown_cell("# ATML PA3 Task 1 — Kaggle controller\n\nEnable GPU and Internet in Kaggle settings. Edit and run the single code cell for one phase per session. It finds a mounted `Assignment-3` root or sparse clones the course repository."),
    nbf.v4.new_code_cell(controller),
    nbf.v4.new_markdown_cell("## Phase order\n\n1. `diagnose`, `prepare`, then **`preflight`**. Inspect `outputs/task1/preflight/report.json`; training requires its pass marker.\n2. Evaluate `baseline`; train each method, resuming at checkpoints 312, 625, 937, 1250 as needed.\n3. Evaluate the six trained conditions; run `weight`, `activations`, and `qualitative` for all seven conditions.\n4. Run `aggregate`, review results, then export and save outputs.\n\nSave `/kaggle/working/outputs/task1` and `/kaggle/working/results/task1` before a session ends. Mount the saved dataset and set `RESTORE_DATASETS` in the next session. Two T4 GPUs have separate VRAM; the trainer uses one GPU."),
]
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
target = Path(__file__).with_name("ATML_PA3_Task1.ipynb")
nbf.write(nb, target)
print(target)
