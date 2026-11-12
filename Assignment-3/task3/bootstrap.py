"""Install missing/incompatible small dependencies; preserve CUDA torch."""
import importlib.metadata as md
from pathlib import Path
import subprocess
import sys

def main():
 import torch
 if not torch.cuda.is_available():raise RuntimeError('Choose Runtime > Change runtime type > T4 GPU first')
 print('[PASS] Preserve installed CUDA PyTorch',torch.__version__,torch.version.cuda,flush=True)
 missing=[]
 for line in Path(__file__).with_name('requirements-colab.txt').read_text().splitlines():
  if not line or line.startswith('#'):continue
  name,version=line.split('==')
  try:current=md.version(name)
  except md.PackageNotFoundError:current=None
  if current!=version:missing.append(line)
  else:print('[SKIP: verified cache] dependency',line,flush=True)
 if missing:
  # Constraints prevent pip from replacing the existing multi-GB framework.
  constraint=Path('/content/task3-torch-constraint.txt');constraint.write_text('torch=='+torch.__version__+'\n')
  print('[START] Install only missing/incompatible dependencies',missing,flush=True)
  subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','-c',str(constraint),*missing],check=True)
 print('[PASS] Dependency setup complete; experiment runs in a fresh child process',flush=True)
if __name__=='__main__':main()
