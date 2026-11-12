"""Run the pinned Task 3 protocol on Kaggle with one live worker per T4."""

from collections import deque
from contextlib import contextmanager
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request

from task3 import run
from task3.data import QuestionReader
from task3.live import CALLS, Generator, run_live
from task3.storage import Monitor, SafeStop, Store, sha256, write_json

SAFE_STOP_EXIT = 75
DISK_RESERVE = 1_500_000_000
MODEL_CACHE_BYTES = 6_183_451_098
official_download = run.download
official_checkpoint_exports = run.checkpoint_exports


class KaggleMonitor(Monitor):
    """Stop at an existing checkpoint boundary if the working disk gets too full."""

    def check(self, estimated_seconds=0):
        super().check(estimated_seconds)
        now = time.monotonic()
        if now >= getattr(self, "next_disk_check", 0):
            if shutil.disk_usage(self.root).free < DISK_RESERVE:
                raise SafeStop("Kaggle working disk below 1.5 GB free; save this output version")
            self.next_disk_check = now + 15


class WorkerMonitor(KaggleMonitor):
    def __init__(self, root, deadline, stop, slot):
        self.stop = stop
        self.slot = slot
        super().__init__(root)
        self.deadline = deadline

    def check(self, estimated_seconds=0):
        if self.stop.is_set():
            raise SafeStop("Other GPU worker stopped; completed calls are saved")
        super().check(estimated_seconds)

    def log(self, kind, message, **fields):
        if "question" in fields and hasattr(self, "question"):
            fields["question"] = self.question
        super().log(kind, f"T4 worker {self.slot}: {message}", **fields)


def check_model_space(cache):
    cached = sum(path.stat().st_size for path in cache.rglob("*")
                 if path.is_file() and not path.is_symlink())
    required = max(0, MODEL_CACHE_BYTES - cached) + DISK_RESERVE
    if shutil.disk_usage(cache).free < required:
        raise SafeStop("Insufficient space to finish the shared model cache; save this output version")


class WorkerStore(Store):
    """Keep each worker's smoke-test writes separate from the parent marker."""

    def __init__(self, root, identity, slot):
        super().__init__(root, identity)
        self.smoke_key = f"gpu_smoke_t4_{slot}"

    def put(self, key, value):
        return super().put(self.smoke_key if key == "gpu_smoke" else key, value)

    def get(self, key):
        return super().get(self.smoke_key if key == "gpu_smoke" else key)


@contextmanager
def model_load_lock(cache, monitor):
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".task3-model-load.lock").open("a") as handle:
        while True:
            monitor.check()
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(1)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def pending_work(ids, store):
    """Resume partial questions on their assigned GPU slot, queue fresh work jointly."""
    assigned = [[], []]
    fresh = []
    for qid in ids:
        present = [store.get(f"live/{qid}/{name}") is not None for name, _ in CALLS]
        if all(present) and store.get(f"live_summary/{qid}") is not None:
            continue
        affinity = store.get(f"gpu_affinity/{qid}")
        if affinity is not None:
            assigned[affinity["slot"]].append(qid)
        elif any(present):
            # Partial questions from the original single-GPU notebook used slot zero.
            assigned[0].append(qid)
        else:
            fresh.append(qid)
    return assigned, fresh


def gpu_worker(slot, device, folder, ids, queue, assigned, checkpoints, identity,
               cache, logs, deadline, stop):
    # Set this before Generator or any model code imports CUDA PyTorch.
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    monitor = WorkerMonitor(logs / f"t4-{slot}", deadline, stop, slot)
    store = WorkerStore(checkpoints, identity, slot)
    try:
        reader = QuestionReader(folder, "math")
        generator = None

        class SessionGenerator(Generator):
            def smoke(self, questions, destination):
                if not getattr(self, "smoke_complete", False):
                    questions = [reader.question(qid)[0]["question_text"] for qid in ids]
                    super().smoke(questions, destination)
                    self.smoke_complete = True

        def get_generator(model_cache, live_monitor):
            nonlocal generator
            if generator is None:
                with model_load_lock(model_cache, live_monitor):
                    check_model_space(model_cache)
                    generator = SessionGenerator(model_cache, live_monitor)
            return generator

        remaining = deque(assigned)
        while True:
            monitor.check()
            qid = remaining.popleft() if remaining else queue.get()
            if qid is None:
                break
            if store.get(f"gpu_affinity/{qid}") is None:
                store.put(f"gpu_affinity/{qid}", {"slot": slot})
            monitor.question = ids.index(qid) + 1
            monitor.log("START", "Live question assigned", question=monitor.question, qid=qid)
            run_live(reader, [qid], store, monitor, cache, generator_factory=get_generator)
    except SafeStop as exc:
        stop.set()
        monitor.log("SAFE STOP", str(exc))
        raise SystemExit(SAFE_STOP_EXIT) from None
    except BaseException:
        stop.set()
        raise
    finally:
        monitor.close()


def run_live_dual(reader, ids, store, monitor, model_cache):
    """Distribute questions; the original generator, seeds, and exports stay intact."""
    assigned, fresh = pending_work(ids, store)
    if not (fresh or any(assigned)):
        monitor.log("SKIP: verified cache", "All live calls and summaries already saved")
        return

    devices = os.environ["TASK3_T4_DEVICES"].split(",")
    if len(devices) != 2 or len(set(devices)) != 2 or not all(devices):
        raise RuntimeError("Two distinct T4 device IDs are required")
    context = mp.get_context("spawn")
    queue = context.Queue()
    stop = context.Event()
    logs = store.root.parent / "gpu-workers"
    for qid in [*fresh, None, None]:
        queue.put(qid)
    workers = []
    started = time.monotonic()
    try:
        for slot, device in enumerate(devices):
            worker = context.Process(
                target=gpu_worker,
                args=(slot, device, reader.folder, ids, queue, assigned[slot], store.root,
                      store.identity, Path(model_cache), logs, monitor.deadline, stop),
            )
            worker.start()
            workers.append(worker)
        while any(worker.is_alive() for worker in workers):
            if any(worker.exitcode not in (None, 0) for worker in workers):
                stop.set()
            for worker in workers:
                worker.join(timeout=0.25)
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        queue.cancel_join_thread()
        queue.close()

    codes = [worker.exitcode for worker in workers]
    smoked = [slot for slot in range(2) if store.get(f"gpu_smoke_t4_{slot}") is not None]
    if smoked:
        store.put("gpu_smoke", {"mode": "Kaggle T4 workers", "verified_worker_slots": smoked})
    write_json(logs / "session.json", {
        "workers": codes,
        "verified_worker_slots": smoked,
        "session_worker_wall_seconds": time.monotonic() - started,
    })
    if any(code not in (0, SAFE_STOP_EXIT) for code in codes):
        raise RuntimeError(f"GPU worker failed (exit codes {codes}); completed calls remain saved")
    if SAFE_STOP_EXIT in codes:
        raise SafeStop("Live workers stopped at the session or disk safety boundary")


def download_release(root, monitor):
    """Recover transient public-release download errors using the same file IDs."""
    from gdown.exceptions import FileURLRetrievalError

    try:
        return official_download(root, monitor)
    except (OSError, FileURLRetrievalError) as exc:
        monitor.log("RETRY", "Trying the direct public-release endpoint", error=repr(exc))
    manifest_path = Path(run.__file__).with_name("release_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    for filename, entry in manifest.items():
        monitor.check()
        destination = root / filename
        if destination.is_file() and sha256(destination) == entry["sha256"]:
            continue
        source = entry["source"]
        url = source if source.startswith("https:") else (
            f"https://drive.usercontent.google.com/download?id={source}&export=download&confirm=t"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".direct.partial")
        try:
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
                remaining = entry["bytes"]
                while remaining:
                    block = response.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        break
                    output.write(block)
                    remaining -= len(block)
                    monitor.check()
                if remaining or response.read(1):
                    raise IOError(f"Unexpected release file size: {filename}")
            if sha256(partial) != entry["sha256"]:
                raise IOError(f"Official release checksum mismatch: {filename}")
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
    write_json(root / "release.json", manifest)
    return official_download(root, monitor)


def checkpoint_exports(store, key, out, names):
    official_checkpoint_exports(store, key, out, names)
    staging = out.parent.parent / "staging" / out.name
    if staging.is_dir():
        shutil.rmtree(staging)


def main():
    run.run_live = run_live_dual
    run.Monitor = KaggleMonitor
    run.download = download_release
    run.checkpoint_exports = checkpoint_exports
    run.main()


if __name__ == "__main__":
    main()
