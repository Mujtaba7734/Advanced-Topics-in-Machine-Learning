"""Build a revision-pinned launcher. The notebook uses only stdlib until bootstrap."""
import argparse
import ast
from pathlib import Path
import nbformat

SETUP='''from google.colab import drive
from pathlib import Path
import datetime, hashlib, json, os, subprocess, sys, threading, time

drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/ATML_PA3_Task3')
ROOT.mkdir(parents=True, exist_ok=True)
LOG = ROOT / 'full_pipeline.log'
_log_lock = threading.RLock()
def log(message):
    line = datetime.datetime.now(datetime.timezone.utc).isoformat() + ' ' + message
    with _log_lock:
        print(line, flush=True)
        with LOG.open('a') as f:
            f.write(line + '\\n')
            f.flush()

def stream(command, cwd=None, env=None):
    log('[START] ' + ' '.join(map(str, command)))
    done = threading.Event()
    def heartbeat():
        while not done.wait(30):
            log('[HEARTBEAT] Command active: ' + ' '.join(map(str, command[:4])))
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
            for line in child.stdout:
                log(line.rstrip())
            code = child.wait()
        if code:
            raise RuntimeError('Command failed with exit code ' + str(code) + ': ' + ' '.join(command))
    except BaseException as exc:
        log('[FAIL] ' + repr(exc))
        failed = {'state': 'failed', 'fully_completed': False, 'error': repr(exc)}
        for name in ('status.json', 'final_verification.json'):
            temp = ROOT / (name + '.partial')
            temp.write_text(json.dumps(failed, indent=2))
            temp.replace(ROOT / name)
        raise
    finally:
        done.set()
        thread.join(timeout=2)
    log('[PASS] Command finished; this is not proof of experiment completion')
(ROOT / 'status.json').write_text(json.dumps({'state': 'not run', 'fully_completed': False}))
log('[START] Task 3 launcher. Logs: ' + str(LOG))
'''
SOURCE='''SOURCE_REVISION = __REVISION__
REPOSITORY = 'https://github.com/Mujtaba7734/Advanced-Topics-in-Machine-Learning.git'
CODE = Path('/content/atml_task3_' + SOURCE_REVISION[:12])
CODE.mkdir(parents=True, exist_ok=True)
stream(['git', 'init', str(CODE)])
stream(['git', '-C', str(CODE), 'fetch', '--depth=1', '--filter=blob:none', REPOSITORY, SOURCE_REVISION])
stream(['git', '-C', str(CODE), 'sparse-checkout', 'set', '--no-cone',
        '/Assignment-3/task3/', '/Assignment-3/tests/test_task3.py', '/Assignment-3/ATML-PA3.pdf'])
stream(['git', '-C', str(CODE), 'checkout', '--detach', 'FETCH_HEAD'])
stream(['git', '-C', str(CODE), 'diff', '--exit-code', 'HEAD', '--', 'Assignment-3/task3', 'Assignment-3/tests/test_task3.py'])
ASSIGNMENT = CODE / 'Assignment-3'
ENV = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONPATH=str(ASSIGNMENT),
           HF_HOME=str(ROOT / 'model_cache'), TASK3_NOTEBOOK_TEE='1', TOKENIZERS_PARALLELISM='false')
log('[PASS] Pinned source commit: ' + SOURCE_REVISION)
stream([sys.executable, '-u', '-m', 'task3.bootstrap'], cwd=str(ASSIGNMENT), env=ENV)
stream([sys.executable, '-u', '-m', 'pytest', '-q', 'tests/test_task3.py'], cwd=str(ASSIGNMENT), env=ENV)
(ROOT / 'status.json').write_text(json.dumps({'state': 'CPU tested', 'fully_completed': False}))
'''
RUN='''log('[START] Official offline experiments, then live T4 experiment. Existing verified results will be reused.')
stream([sys.executable, '-u', '-m', 'task3.run', '--root', str(ROOT)], cwd=str(ASSIGNMENT), env=ENV)
'''
SUMMARY='''verification = ROOT / 'final_verification.json'
if not verification.is_file():
    raise RuntimeError('Missing final_verification.json: completion cannot be claimed.')
result = json.loads(verification.read_text())
if result.get('state') == 'failed':
    raise RuntimeError(result.get('error', 'Pipeline failed'))
if result.get('fully_completed'):
    required_counts = {'offline_datasets': 2, 'live_questions': 50,
                       'live_generation_calls': 250, 'adaptive_policies': 4}
    for name, count in required_counts.items():
        if result.get(name) != count:
            raise RuntimeError('Incomplete count: ' + name)
    artifacts = result.get('artifacts', {})
    if len(artifacts) < 60 or result.get('artifact_count') != len(artifacts):
        raise RuntimeError('Missing required artifact inventory')
    for filename, expected in artifacts.items():
        path = Path(filename)
        if not path.is_file():
            raise RuntimeError('Missing artifact: ' + filename)
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != expected:
            raise RuntimeError('Artifact checksum mismatch: ' + filename)
    log('[COMPLETE] TASK 3 FULLY COMPLETED')
else:
    log('[SAFE STOP] INCOMPLETE — SAVED, RUN ALL AGAIN TO RESUME')
log('State: ' + result.get('state', 'unknown'))
log('Offline datasets: ' + str(result.get('offline_datasets', 0)) + '/2')
log('Live questions: ' + str(result.get('live_questions', 0)) + '/50; generation calls: '
    + str(result.get('live_generation_calls', 0)) + '/250')
log('Adaptive policies: ' + str(result.get('adaptive_policies', 0)) + '/4; plots: '
    + str(result.get('plots', 0)) + '; tables: ' + str(result.get('tables', 0))
    + '; total artifacts: ' + str(result.get('artifact_count', 0)))
log('Latest checkpoint: ' + str(result.get('latest_checkpoint', ROOT / 'latest_run.json')))
log('Saved Drive path: ' + str(ROOT))
log('fully_completed = ' + str(result.get('fully_completed', False)))
'''

def build(revision,destination):
 if len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):raise ValueError('Require exact commit SHA')
 notebook=nbformat.v4.new_notebook(metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python'),accelerator='GPU',colab=dict(name='ATML_PA3_Task3_RunAll.ipynb',gpuType='T4')))
 notebook.cells=[nbformat.v4.new_markdown_cell('''# ATML PA3 Task 3 — Run All on one T4

1. Select **Runtime → Change runtime type → T4 GPU**.
2. Select **Runtime → Run all**, then approve Google Drive access.

No configuration edits or uploads are required. Logs print live and are saved to `MyDrive/ATML_PA3_Task3/`. A safe stop means **run this same notebook again** in the next session; completed questions/calls are checksum-verified and reused. Five hours is a session budget, **not a runtime promise**.

The notebook runs CPU tests, the official offline protocol, a real FP16 T4 memory smoke, and the fixed 50-question experiment. It reports full completion only after validating all required evidence. Initial implementation verification is recorded in the repository; a real full Colab run is still required.
''')]
 for source in (SETUP,SOURCE.replace('__REVISION__',repr(revision)),RUN,SUMMARY):
  ast.parse(source);notebook.cells.append(nbformat.v4.new_code_cell(source))
 nbformat.validate(notebook);nbformat.write(notebook,destination)
 reread=nbformat.read(destination,as_version=4);nbformat.validate(reread)
 assert all(c.execution_count is None and not c.outputs for c in reread.cells if c.cell_type=='code')
 print('[PASS] nbformat, Python syntax, clean outputs, sequential workflow:',destination)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--revision',required=True);p.add_argument('--output',default=str(Path(__file__).parents[1]/'colab/ATML_PA3_Task3_RunAll.ipynb'));args=p.parse_args();build(args.revision,args.output)
