"""Build the Kaggle launcher for the existing revision-pinned Task 3 pipeline."""

import ast
from pathlib import Path

import nbformat

from task3.build_notebook import SETUP, SOURCE, SUMMARY

REVISION = "b370e572a01f5b0c3cd4d12e0195314be647093f"

INTRO = """# ATML PA3 Task 3 on Kaggle

1. Import this notebook into Kaggle. Enable **Internet** and select **GPU T4 ×2**. Each T4 handles independent live questions with the original FP16 model and batch size 1. All strategies for a question stay on the same GPU slot.
2. Select **Save Version → Save & Run All** with the GPU enabled. For an interactive run, save the session outputs before stopping it.
3. If the run stops before completion, attach the saved version through **Add Input → Notebook Output Files**, then run the notebook again. Attach only one Task 3 output version at a time. The launcher automatically restores its files and the pipeline checks checkpoint identity and checksums.

Logs, outputs, and checkpoints are written under `/kaggle/working/ATML_PA3_Task3/`. Both workers share a single on-disk model cache. The launcher budgets 12 GB for Task 3, checks for 1.5 GB of additional headroom, and stops at a checkpoint boundary if the working disk runs low. Only saved Kaggle outputs persist across sessions. The five-hour session budget is a safe-stop setting. Total runtime has not been measured on Kaggle.
"""

RESTORE = '''import shutil

TASK3_PEAK_BUDGET = 12_000_000_000
KAGGLE_OUTPUT_LIMIT = 20_000_000_000
DISK_RESERVE = 1_500_000_000

def tree_bytes(path):
    return sum(item.stat().st_size for item in path.rglob('*')
               if not item.is_symlink() and item.is_file())

def check_storage(root, incoming=0):
    current = tree_bytes(root)
    other = max(0, tree_bytes(root.parent) - current)
    required = max(TASK3_PEAK_BUDGET - current, incoming)
    free = shutil.disk_usage(root).free
    if other + current + required + DISK_RESERVE > KAGGLE_OUTPUT_LIMIT or free < required + DISK_RESERVE:
        raise RuntimeError(
            f'Insufficient Kaggle space for Task 3: {free / 1e9:.2f} GB free, '
            f'{other / 1e9:.2f} GB of other working files. Use a clean working directory.'
        )
    log(f'[STORAGE] {free / 1e9:.2f} GB free; {current / 1e9:.2f} GB Task 3 data already present.')

def restore_saved_output(root, input_dir):
    """Fill missing files from one attached Task 3 version, preserving current work."""
    candidates = []
    for current, directories, _ in os.walk(input_dir):
        folder = Path(current)
        if folder.name == root.name and any((folder / name).is_dir() for name in ('release', 'runs')):
            candidates.append(folder)
            directories[:] = []
        elif len(folder.relative_to(input_dir).parts) >= 7:
            directories[:] = []
    if len(candidates) > 1:
        raise RuntimeError('Attach only the Task 3 output version you want to resume.')
    if not candidates:
        log('[START] No attached Task 3 output version found.')
        check_storage(root)
        return
    source = candidates[0]
    log('[START] Restoring from ' + str(source))
    copies = []
    for name in ('release', 'runs', 'model_cache'):
        parent = source / name
        if not parent.is_dir():
            continue
        for current, directories, files in os.walk(parent):
            directories[:] = [directory for directory in directories
                              if directory not in ('staging', '.locks')]
            for filename in files:
                path = Path(current) / filename
                if filename.endswith(('.partial', '.incomplete', '.lock')) or not path.is_file():
                    continue
                target = root / path.relative_to(source)
                if target.exists():
                    continue
                copies.append((path, target))
    check_storage(root, incoming=sum(path.stat().st_size for path, _ in copies))
    for path, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.restore.partial')
        shutil.copyfile(path, temporary)
        temporary.replace(target)
    log('[PASS] Attached files restored. The worker will validate them before reuse.')

restore_saved_output(ROOT, Path('/kaggle/input'))
'''

GPU = '''# Workers will each narrow this allocation to one T4 before importing PyTorch.
visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1')
ENV['CUDA_VISIBLE_DEVICES'] = visible
ENV['TASK3_T4_DEVICES'] = ','.join(device.strip() for device in visible.split(','))
ENV['HF_HUB_DISABLE_XET'] = '1'
ENV['PIP_NO_CACHE_DIR'] = '1'
stream([sys.executable, '-u', '-c', """
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 2 or not all('T4' in name for name in names):
    raise RuntimeError(f'Enable GPU T4 x2 in Kaggle settings. Detected {names}.')
print(f'[PASS] Both GPUs available: {names}; installed PyTorch {torch.__version__}')
"""], cwd=str(ASSIGNMENT), env=ENV)
'''

DEPENDENCIES = '''import importlib.metadata as metadata

missing = []
for requirement in (ASSIGNMENT / 'task3/requirements-colab.txt').read_text().splitlines():
    if not requirement or requirement.startswith('#'):
        continue
    package, version = requirement.split('==', 1)
    try:
        installed = metadata.version(package)
    except metadata.PackageNotFoundError:
        installed = None
    if installed != version:
        missing.append(requirement)
if missing:
    constraint = CODE / 'torch-constraint.txt'
    constraint.write_text('torch==' + metadata.version('torch') + '\\n')
    stream([sys.executable, '-u', '-m', 'pip', 'install', '--disable-pip-version-check',
            '--prefer-binary', '-c', str(constraint), *missing], env=ENV)
else:
    log('[SKIP] All pinned dependencies are already installed.')
stream([sys.executable, '-u', '-m', 'pytest', '-q', 'tests/test_task3.py'],
       cwd=str(ASSIGNMENT), env=ENV)
'''

PORTABLE_CACHE = '''def prepare_saved_output(root):
    """Materialize Hugging Face snapshot links without keeping duplicate model blobs."""
    from hashlib import file_digest

    model_cache = root / 'model_cache'
    for manifest_file in model_cache.glob('*-verified.json'):
        manifest = json.loads(manifest_file.read_text())
        model_dir = model_cache / ('models--' + manifest['model'].replace('/', '--'))
        snapshot = model_dir / 'snapshots' / manifest['revision']
        blob_dir = (model_dir / 'blobs').resolve()
        links = []
        for filename, expected in manifest['files'].items():
            path = snapshot / filename
            if not path.is_symlink():
                continue
            blob = path.resolve()
            if not blob.is_relative_to(blob_dir) or not blob.is_file():
                raise RuntimeError('Invalid cached model link: ' + str(path))
            with blob.open('rb') as handle:
                if file_digest(handle, 'sha256').hexdigest() != expected:
                    raise RuntimeError('Cached model checksum mismatch: ' + str(path))
            links.append((path, blob))
        for path, blob in links:
            temporary = path.with_name(path.name + '.portable.partial')
            temporary.unlink(missing_ok=True)
            os.link(blob, temporary)
            temporary.replace(path)
        for blob in {blob for _, blob in links}:
            # Keep blobs still referenced by any other snapshot in this cache.
            if not any(link.is_symlink() and link.resolve() == blob
                       for link in (model_dir / 'snapshots').rglob('*')):
                blob.unlink()
        if links:
            log('[PASS] Verified model snapshot prepared as regular Kaggle output files.')
    # These are reconstructible copies, never the release, exports, or checkpoints.
    for temporary in [root / 'local_ssd', *root.glob('runs/*/staging')]:
        if temporary.is_dir() and not temporary.is_symlink():
            shutil.rmtree(temporary)

DRIVER = CODE / 'task3_kaggle_driver.py'
DRIVER.write_text(__DRIVER__)
release = ROOT / 'release'
release.mkdir(exist_ok=True)
local_release = ROOT / 'local_ssd'
if not local_release.exists() and not local_release.is_symlink():
    local_release.symlink_to(release, target_is_directory=True)
try:
    log('[START] Offline analysis, then live questions on two T4 workers.')
    stream([sys.executable, '-u', str(DRIVER), '--root', str(ROOT)], cwd=str(ASSIGNMENT), env=ENV)
finally:
    if local_release.is_symlink() and local_release.resolve() == release:
        local_release.unlink()
    prepare_saved_output(ROOT)
    log(f'[STORAGE] Task 3 saved output: {tree_bytes(ROOT) / 1e9:.2f} GB.')
    log('[KAGGLE OUTPUT] Save this version with output files before ending the session.')
'''


def build(destination):
    setup = SETUP.replace("from google.colab import drive\n", "").replace(
        "drive.mount('/content/drive')\nROOT = Path('/content/drive/MyDrive/ATML_PA3_Task3')",
        "ROOT = Path('/kaggle/working/ATML_PA3_Task3')",
    )
    source = SOURCE.split("stream([sys.executable, '-u', '-m', 'task3.bootstrap']", 1)[0]
    source = source.replace("__REVISION__", repr(REVISION)).replace(
        "'/content/atml_task3_'", "'/tmp/atml_task3_'"
    )
    driver = Path(__file__).with_name("task3_dual.py").read_text()
    run = PORTABLE_CACHE.replace("__DRIVER__", repr(driver))
    summary = SUMMARY.replace(
        "INCOMPLETE — SAVED, RUN ALL AGAIN TO RESUME",
        "INCOMPLETE — SAVE KAGGLE OUTPUTS AND ATTACH THEM TO RESUME",
    ).replace("Saved Drive path", "Kaggle output path")
    notebook = nbformat.v4.new_notebook(
        metadata=dict(
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            language_info=dict(name="python"),
        )
    )
    notebook.cells = [nbformat.v4.new_markdown_cell(INTRO)]
    for code in (setup + RESTORE, source + GPU + DEPENDENCIES, run, summary):
        ast.parse(code)
        notebook.cells.append(nbformat.v4.new_code_cell(code))
    nbformat.validate(notebook)
    nbformat.write(notebook, destination)


if __name__ == "__main__":
    build(Path(__file__).with_name("ATML_PA3_Task3_Kaggle_RunAll.ipynb"))
