"""Fresh subprocess setup: leave compatible large Torch installation in place."""
import importlib.metadata as md
from pathlib import Path
import subprocess
import sys

def main():
    from packaging.requirements import Requirement
    from packaging.version import Version
    try: version=Version(md.version('torch').split('+')[0])
    except md.PackageNotFoundError: version=None
    if version is None or not Version('2.6')<=version<Version('3'):
        raise RuntimeError('Colab must supply CUDA-enabled PyTorch >=2.6,<3. No automatic multi-GB Torch replacement.')
    missing=[]
    for line in Path(__file__).with_name('requirements-colab.txt').read_text().splitlines():
        line=line.strip()
        if not line or line.startswith('#'): continue
        req=Requirement(line)
        try: compatible=md.version(req.name) in req.specifier
        except md.PackageNotFoundError: compatible=False
        if not compatible: missing.append(line)
    if missing:
        print('[SETUP] Installing only missing/incompatible packages:',missing,flush=True)
        # A Torch constraint prevents pip from upgrading an already compatible GPU stack.
        constraint=Path('/tmp/atml_task2_torch_constraint.txt')
        constraint.write_text('torch=='+md.version('torch')+'\n')
        subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','-c',str(constraint),*missing],check=True)
    else: print('[SETUP] Compatible packages already installed; no installation required.',flush=True)
    subprocess.run([sys.executable,'-c','import torch, transformers, peft; from transformers import Qwen3ForCausalLM, SiglipVisionModel; print(torch.__version__, transformers.__version__, peft.__version__)'],check=True)

if __name__=='__main__': main()
