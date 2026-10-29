# Task 1 engineering audit — 29 October 2026

`ATML-PA3.pdf` is authoritative. This audit changes validation, recovery, and the Kaggle controller without changing the experiment settings in `configs/task1.yaml`. No full training or official benchmark result was generated locally.

## Defects and corrections

| Defect | Cause and affected code | Correction | Verification |
|---|---|---|---|
| Qwen assistant prefix exception | `prepare_data.encode_example` compared independently rendered chat templates as though one were a token prefix of the other. Qwen's non-thinking generation header has its own layout. | Render the non-thinking user generation prompt once, tokenize it and the supplied response separately, append `<|im_end|>`, decode and check the complete sequence, and mask the prompt. | BatchEncoding-shaped and boundary tests pass locally; **real Qwen tokenizer remains NOT RUN** locally and is mandatory in Kaggle preflight. |
| `BatchEncoding + list` | Tokenizer call result was treated as a flat list. | Explicitly unwrap `input_ids` from BatchEncoding and reject unexpected shapes. | Local BatchEncoding-shaped test PASSED; real tokenizer NOT RUN. |
| Wrong Kaggle repository path | Old notebook assumed `/kaggle/input/atml-pa3` or `/kaggle/working/ATML-PA3`. | One self-contained cell discovers a mounted `Assignment-3` folder or sparse clones the GitHub repository into `/kaggle/working/atml-repo/Assignment-3`. Source hashes reject a stale GitHub checkout. | Notebook syntax and schema validation PASSED; actual Kaggle mount/clone NOT RUN. |
| Training could start with no real integration proof | No preflight gate existed. | Added `task1.preflight` with real tokenizer, data, five method, checkpoint, harness, generation, plotting, and resource checks. Training requires a matching pass marker and report. | Local CPU subset PASSED; GPU integration NOT RUN. |
| Existing run config could be overwritten before resume validation | Training wrote `config.json` before checking the recovery state. | Write config only after recovery validation. Fall back to the newest valid recovery if `latest.pt` is absent or damaged; refuse a silent restart if outputs exist without recovery. | Unit checkpoint and interrupted versus uninterrupted short run PASSED; Kaggle recovery NOT RUN. |
| Full FT displacement reference could be lost | The full method needs `W0` for the required midpoint/final displacement. | Keep a full CPU reference for Full FT; use pinned CPU storage only for L2-SP transfers. | Small L2-SP reference test PASSED; full-model displacement NOT RUN. |
| AdamW peak memory could be larger than needed | CUDA foreach optimizer may use extra tensor-list memory. | Use `foreach=False`, with identical AdamW update arithmetic. | Local short optimizer/resume test PASSED; T4 memory NOT RUN. |
| Saved evaluation could silently belong to another checkpoint | Reuse was based on sample count alone. | Fingerprint the config and inference checkpoint; reject stale result files. Load the model only if a task is pending. | Static/unit checks PASSED; real harness integration NOT RUN. |
| HellaSwag and ARC-Easy per-sample prediction could disagree with the primary metric | The saved prediction used raw log likelihood argmax while the PDF's primary score is harness `acc_norm`. | Save both raw and character-length-normalized choices, matching pinned harness task code. | Synthetic choice tests PASSED; real harness sample structure NOT RUN. |
| Qualitative partial output could belong to another model | Existing rows checked prompt SHA but no checkpoint identity. | Save and validate the condition fingerprint on generation/resume. | Structural 24-prompt and 168/42 fixture tests PASSED; real generation NOT RUN. |
| Diagnostics repeatedly loaded large checkpoints | Each layer/condition pair loaded the full checkpoint again. | Load one condition checkpoint for all three layers and validate saved probe IDs and baseline activation cache provenance. | Metric equations PASSED on small tensors; real checkpoint diagnostics NOT RUN. |
| Prepared data could be reused without validation | Preparation always reselected even if complete data existed. | Validate saved SHA, count, order, and model before reuse; validate real source schemas and write files atomically. | Synthetic selection/integrity tests PASSED; actual 40,000-row selection NOT RUN. |

## Verification status

| Check | Status | Evidence or reason |
|---|---|---|
| Python syntax (`compileall`) | PASSED | All `task1`, `common`, `kaggle`, and `tests` modules compile. |
| Local unit suite | PASSED | 22 tests, including small Qwen3 architecture loss/gradient, the shared accumulated update after Adam initialization, adapter projection/merge, independent math, checkpoint/RNG restoration, secondary-GPU AdamW equivalence and resume, short interrupted/resumed optimizer equivalence, CPU-only preflight gate, official YAML structure, and synthetic plots. |
| CPU-only preflight | PASSED | Prompt schema, local unit suite, and plotting/serialization fixture reported PASSED. |
| Real Qwen3-0.6B tokenizer and weights | NOT RUN | No local model cache or GPU. Kaggle `preflight` loads exact pinned versions and saves tokenization details. |
| Actual MetaMathQA and GSM8K data | NOT RUN | No local dataset cache/network. Kaggle `prepare` must save exactly 40,000 examples; `preflight` checks split schemas, fingerprints, selected rows, and real tokenization. |
| Five actual Qwen3 method optimizer and recovery checks | NOT RUN | GPU required. Kaggle preflight now performs three consecutive 32-example updates per method using the longest prepared sequences, with gradient accumulation 16, initialized Adam moments, checkpoint save/reload, inference-mode verification, merge/projection checks, and GPU headroom logging. |
| Real harness task subset and one qualitative generation | NOT RUN | `lm-eval` and GPU unavailable locally. Kaggle preflight uses **two examples per task only as an interface smoke test**; these are never written as official results. |
| Full 1,250-update training and all seven full evaluations | NOT RUN | Explicitly deferred to Kaggle after preflight. |
| Actual Kaggle notebook mount, network, pip install, and save/restore | NOT RUN | Requires a Kaggle session. |

## Execution and recovery

1. Mount the current audited `Assignment-3` folder as a Kaggle input and open `kaggle/ATML_PA3_Task1.ipynb`. The notebook can sparse clone the public GitHub source, but its embedded hashes stop if that checkout is older than this local audit. The current changes were **not pushed**.
2. In the one controller cell, run `diagnose`, then `prepare`, then `preflight`. Stop if any preflight check fails. Inspect `/kaggle/working/outputs/task1/preflight/report.json` and `data/preflight_pass.json`.
3. Run `evaluate` for `baseline`. Run `train` separately for `full`, `l2sp`, `lora4`, `lora16`, and `oplora16`. For disk-limited sessions, use stop points 312, 625, 937, and 1250 without changing the scheduler or data order.
4. Save `/kaggle/working/outputs/task1` and `/kaggle/working/results/task1` as a Kaggle version/dataset before a session ends. In the next session, attach that output dataset and set `RESTORE_DATASETS` and `METHOD` in the same controller cell. The restore code copies data, pass report, results, the chosen method's latest valid recovery or newest recovery file, log, and inference files. The trainer restores model, AdamW, scheduler, scaler, RNG, skipped count, and next example position; logs after the recovered update are trimmed.
5. Evaluate the six trained conditions; run `weight`, `activations`, and `qualitative` for all seven conditions; run `aggregate`. Use `export` only when working disk has room for the ZIP. The notebook restores all inference files for analysis phases.

## Assignment compliance and limits

The [traceability table](task1_traceability.md) maps each PDF requirement to implementation and expected output. `task1/protocol.py` rejects changes to 40,000 examples, 2,048 tokens, batch 2 × accumulation 16, 1,250 updates, FP16 computation, seed 6304, required learning rates, targets, checkpoint updates, diagnostic layers, and generation caps.

The two Kaggle T4 devices have independent ~16 GiB VRAM. Full FT has roughly 2.4 GiB FP32 weights and 2.4 GiB gradients on GPU 0; FP32 mirror weights (~2.4 GiB) and Adam moments (~4.8 GiB) reside on GPU 1. L2-SP adds a full pinned host reference and transfers; OPLoRA uses GPU 1 for its FP32 SVD workspace. Checkpoints save the model and Adam moments, reconstructing mirror weights on resume. The preflight measures three consecutive worst-length updates, checkpoint save/reload, and a 1 GiB peak reserved-memory margin. **T4 fit remains NOT RUN on this CPU-only machine.** If either GPU fails, a single [40 GB or 80 GB A100-class accelerator](https://images.nvidia.com/data-center/a100/a100-datasheet.pdf) with ample host RAM and storage is a realistic alternative that still requires preflight; full FSDP/ZeRO sharding would need separate implementation and validation. Do not change the assignment batch, precision, lengths, or data size to make it fit.

## 30 October memory-failure addendum

The Kaggle report showed CUDA OOM during the old preflight's second, post-reload forward pass: GPU 0 had only 22.81 MiB free for a 20 MiB allocation. The old preflight retained gradients after `optimizer.step()`, kept state and loaded checkpoint objects alive, and built an autograd graph for a verification forward. It also tested only one two-example step, so it did not model training after Adam state initialization or gradient accumulation. The actual trainer similarly left gradients allocated until the next update and while saving a checkpoint.

The new shared `execute_update()` path clears gradients immediately after a successful step, deletes each microbatch's graph-bearing tensors, and is used by both training and preflight. The reload check clears gradients, frees checkpoint objects, and uses `torch.inference_mode()`. Full FT and L2-SP use `SecondaryGPUAdamW`, which delegates to PyTorch AdamW on GPU 1 and copies unscaled, clipped gradients from GPU 0. A CPU reference test matches ordinary AdamW exactly for multiple scheduled updates, recovery, and native-state migration. The report includes per-update duration and a conservative full-run projection from the longest examples; the offload copies large tensors between GPUs and may be slow. The preflight has not yet run on the real T4 pair, so the reported OOM is **not claimed resolved on Kaggle** until it completes.

GSM8K `truncation_flag` is a lower-bound heuristic from re-tokenizing harness text, because the pinned harness sample does not expose the generation stop reason. The output labels it `retokenized_text_lower_bound`; it is not an exact stop-at-cap indicator. The real harness and task sample structure remain a Kaggle preflight gate. The professor's PDF report and interpretation remain student-written.
