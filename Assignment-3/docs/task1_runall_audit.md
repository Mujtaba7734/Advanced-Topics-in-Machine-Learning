# Task 1 Run All notebook audit

## Scope and status

The generated notebook has 18 ordered, self-contained stage cells. Each cell starts a new Python process using the notebook's interpreter, checks the embedded source-manifest hash, verifies pinned packages and essential imports, and invokes one stage in `kaggle/runall.py`. The source manifest hashes the Task 1 Python modules, configuration, requirements, official prompt YAML, assignment PDF, traceability document, and notebook generator. A mismatched source fails before any experiment runs.

The previous baseline failure occurred after a successful GPU preflight because the old notebook's one-time package-install flag did not guarantee that `lm_eval` existed in the interpreter used by a later session or phase. The new runner probes a clean child interpreter before each dependent stage and installs only missing or incompatible pinned packages. It also retries the import probe after removing a broken optional TorchVision installation. Each notebook cell runs in a fresh child process, so a package imported earlier in the notebook kernel cannot mask an installation problem.

| Check | Status | Evidence and limit |
|---|---|---|
| Source syntax and notebook JSON/cell syntax | PASS | `compileall`, `nbformat` validation and compilation of all 18 code cells. |
| Source revision and embedded notebook hash | PASS | Local test rejects a changed source file; notebook hash matches the generated manifest. Two consecutive generations produced the same notebook SHA256. |
| Local unit and regression suite | PASS | 36 tests pass, including earlier Task 1 objective, optimizer, and resume tests. |
| Isolated interpreter bootstrap | PASS | `python -I` imports the standard-library runner and verifies the source manifest without project packages on the import path. |
| Stage order, method selection, and stage-state persistence | PASS | All 18 stages executed under lightweight mocks in notebook order; five training method arguments were verified. |
| Prepared-data reuse | PASS | A verified data fixture skips preparation. The real data loader validates count, sequence hash and source order; the stage validates fixed probe IDs. |
| Recovery and checkpoint identity | PASS | Synthetic recovery file validates method, data hash, config, step, and data position. Mounted checkpoint hash clears a pending durable-backup gate; existing working result is preserved. A stale `latest.pt` alias is advanced to the verified newest recovery. |
| Dependency repair logic | PASS | Mocked missing or broken `lm-eval` and broken TorchVision trigger targeted install/uninstall and a clean subprocess reprobe. Actual package installation on Kaggle is **NOT RUN** locally. |
| Checkpoint disk/time planner and compact export | PASS | Fixtures cover insufficient disk/time, recovery exclusion from ZIP, manifest hashes and checkpoint kind, and reuse of an unchanged ZIP. |
| Final completion guard | PASS | Missing computational artifacts yield `INCOMPLETE`; the report never declares Task 1 computationally complete. |
| Real Qwen3 bounded evaluation smoke, full benchmarks and five full training runs | NOT RUN | The local machine has CPU PyTorch and lacks Kaggle T4 GPUs, model/cache and benchmark datasets. The bounded smoke uses the real `task1.evaluation.evaluate()` path with `limit=2` for each task, but must run in Kaggle to validate its model and harness interfaces. |
| Real Kaggle Run All / Resume All and Save Version restoration | NOT RUN | Requires a fresh Kaggle T4 ×2 session and a mounted output dataset. The user's earlier successful GPU preflight is reported, not independently reproduced in this audit. |

## Recovery and resource boundary

The controller chooses the furthest required checkpoint (312, 625, 937, or 1250) that fits measured preflight checkpoint size, free disk, projected time and a 2 GiB disk margin. The trainer also performs its independent host RAM and disk checks. A partial training stage records the checkpoint SHA256 and blocks dependent stages until **Kaggle Save Version** has produced a mounted output dataset with that checkpoint. The compact ZIP excludes every `.pt`, `.safetensors`, and `.bin` file; its manifest lists full recovery and inference files with sizes and hashes. It does not claim the files are durably saved. The notebook preserves prepared data, completed results and checkpoints; a stale preflight pass marker is replaced only when the real preflight reruns.

The time planner uses a conservative eight-hour session budget and the preflight's three-update timing projection. Kaggle may impose a shorter limit or have slower sustained performance. The only validation of sustained T4 memory and the baseline harness is a new Kaggle execution; no local result is labeled as a GPU pass. If the real GPU preflight fails, training remains blocked. If the required checkpoint cannot fit disk or host RAM, use hardware/storage with sufficient capacity rather than changing the assignment's batch, length, precision, optimizer, or update count.

## One required Kaggle verification

Import `kaggle/ATML_PA3_Task1_RunAll.ipynb`, enable T4 ×2 and Internet, attach the audited `Assignment-3` folder and any saved Task 1 output dataset, and click **Run All**. Read the short stage messages and `results/task1/summary/final_verification.json`. At a planned partial checkpoint, use **Save Version**, attach the saved output in the next session, and click **Run All** again. If the first real bounded baseline smoke fails, that cell stops before any full benchmark and records the exact exception. The written scientific report is separate and remains the student's work.
