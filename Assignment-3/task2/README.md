# PA3 Task 2 — single-T4 Colab curriculum

Open `../colab/ATML_PA3_Task2_RunAll.ipynb`, select a T4 GPU, choose **Runtime → Run all**, and authorize the normal Google Drive mount when prompted.

The notebook launches versioned Python modules in a fresh subprocess. No cells contain hidden training state. The immutable implementation commit is embedded in the launcher. This directory is independent of Task 1: no Task 1 code, tests, dependencies, caches or outputs are changed.

## Verification status

CPU unit/integration tests are executed during development and again before every Colab run. They use small, randomly initialized Qwen architectures where model execution is needed. These are **not** GPU or pretrained-model performance tests. See `../docs/task2_validation.json` for the actual development checks. A full T4 run is required to establish memory, convergence, timing and final results. There is no five-hour completion claim.

The runtime itself records distinct `not run`, `CPU tested`, `GPU smoke tested`, `partially trained`, and `fully completed` states. The last state requires all 4,500 successful main optimizer updates, 72,000 presentations, all development evaluations and all final conditions. CPU fixtures never supply scientific results.

## First session and subsequent sessions

1. Retain compatible Colab CUDA PyTorch (>=2.6,<3); install only missing/incompatible small dependencies in `requirements-colab.txt`. Run the pipeline in a new Python process so dependency changes need no notebook-kernel restart. If the GPU stack itself is incompatible, fail clearly rather than silently reinstall it.
2. Resolve original Hugging Face model revisions once and save immutable SHA IDs. Download official COCO/VQA annotation archives, select examples with seed 6304, and save source checksums, IDs, references, and fixed mismatched-image mappings. Download only selected images, not full COCO image archives.
3. Cache frozen SigLIP final patch features in 64-image FP16 shards. Test the final-layernorm call count and processor determinism. Use the model's already-normalized `last_hidden_state`; never normalize it again. Validate each shard before use, and reuse local files when their checksum matches. Remove SigLIP from CUDA before Qwen loading.
4. Stress all three training stages on real selected features with 80-question/64-target-token fixtures, three consecutive official 8×2 updates, dynamic scaling, gradient gates, memory accounting and interrupted-vs-continuous resume checks. Also benchmark 16×1 in isolation; it never changes the mandatory recipe. Synthetic stress text is used only for preflight.
5. Restore fresh original-model initialization after benchmarks, calibrate the connector using the first 64 selected training captions, and train the exact curriculum. Restore a compatible recovery automatically if present. Every new runtime reruns CUDA checks because the actual allocation can differ.
6. Stop at the configured session budget minus a 15-minute reserve, checkpoint and verify, then resume with the next Run All. Feature caching resumes shard by shard. Final evaluation resumes condition by condition. A completed, checksum-verified run returns immediately.

Permanent root: `/content/drive/MyDrive/ATML_PA3_Task2/`. `runs/<source-and-protocol-id>/` contains results, checkpoints, manifests and logs; `data/`, `features/`, and `models/` hold reusable assets. Local SSD: `/content/atml_pa3_task2/`. No Task 1 directories are referenced. Full run identity additionally binds selected data, model revisions, dependency versions and sampling-plan hashes. Version differences fail explicitly instead of silently resuming a different numerical environment.

Checkpoint files include connector/adapters, calibration scale, AdamW states, the closed-form scheduler's stage/position, GradScaler, Python/NumPy/Torch/CUDA RNG, exact plan hash and update cursor, and accumulated per-example logs. Complete local snapshots are copied, checksummed, deserialized from Drive, and only then published through `latest.json`. The previous verified snapshot remains until the next succeeds. Incomplete transfer files never become latest. Unsaved work after the last verified checkpoint can replay after an unexpected disconnect; completed durable updates are never double-counted.

## Scientific choices fixed before results

- Original frozen FP16 Qwen3-0.6B; FP32 connector and LoRA parameters; FP16 autocast and dynamic scaling; no quantization/BF16. Stage 1 has no adapters. Stage 2 introduces q/k/v/o LoRA rank 16, alpha 32, dropout .05, no bias; stage 3 continues them.
- Native thinking-disabled chat template, visual tokens inside the user message before the question, 80/64/16 content-token caps. Train on answer plus end-of-turn; answer NLL excludes end-of-turn. No-image controls omit all 196 embeddings and reconstruct positions/masks.
- AdamW weight decay .01 on trainable parameters, default betas (.9,.999), epsilon 1e-8. Per-stage warmup uses **ceil(5% updates)** = 113/75/38 successful updates, then cosine to .1 of peak. Stage optimizer/scaler are initialized at each stage; model and adapter weights continue unchanged. All choices are recorded.
- Stage 1: two independent seed-6304 shuffled passes, each containing all 18,000 caption examples exactly once. Later stages randomly draw full pairs (or two independent caption examples) using the required probabilities; no quotas enforced on realized mixture counts. Sampling is precommitted in `sampling_plan.json`.
- Grounding evaluation uses equal presence/counting pair counts (12/12 dev; 32/32 final), since their allocation is unspecified. Training pools have 3,072 pairs per type and allow image reuse. Presence uses annotation absence, not a claim of verified visual absence. Counting counts all instance records, including crowd records as one, and restricts counts to 1–4. This convention is explicit in the selected-data manifest. Annotation incompleteness/crowds remain limitations of these generated labels.
- Natural pairs require identical original question strings, distinct images, different normalized consensus answers, matching answer types, and >=8/10 agreement on each side. Training also uses unique images across selected natural pairs (an allowed stricter selection), and evaluation images are disjoint across all suites. Original consensus strings are training targets; all ten annotations are kept.
- Local VQA scoring applies official punctuation/digit/article/contraction normalization consistently to every answer, then the assignment's leave-one-annotator-out formula. It is not the shortcut `min(matches/3,1)`.
- Final decoding is greedy, batch 8, max 16 new tokens. Fixed unrelated-image mapping excludes both original pair images. References never change under controls. Confidence intervals resample complete pairs 4,000 times; condition differences share pair indices. Empty generations stay in denominators.
- Qualitative examples are the first four pair IDs per suite/type in lexical order. JSON plus self-contained aligned-image HTML are generated without report prose.
- Optional 49-token compression is **not implemented/enabled**. The mandatory launcher always uses 196 tokens. A future optional experiment must be a separately identified complete recipe, never a memory fallback.

## Memory and throughput

The transformer processes the complete visual/text sequence. Only supervised hidden positions are projected through the vocabulary head, in checkpointed chunks of 32. This computes the same exact per-token cross-entropy and gradients without materializing vocabulary logits for 196 masked visual positions. CPU tests compare it against full vocabulary projection. FP32 losses are averaged per example, then divided by 16 across accumulation. Chunking is an allocation optimization, not a changed objective.

Feature batches are read from local NumPy memory maps and transferred through pinned CPU buffers. Tokenization is cached. Small hot logs remain local; the complete log history is included in every recovery checkpoint, with heartbeat publication every 25 updates. This avoids 4,500 Drive file transactions in the training hot path. Measured CUDA update times, cache/copy costs, evaluation costs and peak allocated/reserved bytes populate timing and memory JSON. Runtime projections are explicitly extrapolations; unknown costs and actual completion status are separate.

## Commands

From `Assignment-3/`:

```bash
python -m task2.bootstrap
python -m pytest -q tests/test_task2.py
python -m task2.run
# Same full-data/real-CUDA gates, no main training:
python -m task2.run --smoke-only
```

Session duration and reserve may be adjusted via `--session-hours` and `--reserve-minutes`; they do not alter scientific settings. Changing the scientific protocol or source gives a new run ID. Never edit hyperparameters mid-run.

## Outputs

`preflight.json`, `source_manifest.json`, `model_revisions.json`, selected-data and tokenized-example manifests, `sampling_plan.json`, `run_identity.json`, `environment.json`, `calibration.json`, per-stage CUDA records, `checkpoint_manifest.json` under `checkpoints/`, `heartbeat.json`, `timing.json`, `memory.json`, stage weights, `training_logs.json`, `dev-*.json`, all eight raw final condition files, `metrics.json`, `learning_curves.png`, qualitative JSON/HTML and `final_verification.json`.

No experiment artifacts or model/data binaries belong in Git. The assignment prohibits AI-authored final report prose: this implementation supplies code and measured evidence only.

## Attribution and sources

Specification: `../ATML-PA3.pdf`, Task 2 pp. 6–10. VQA normalization table/rules adapted from [GT-Vision-Lab/VQA](https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py). Models use Hugging Face Transformers and PEFT. Data: [COCO](https://cocodataset.org/), [official VQA downloads](https://visualqa.org/download.html). COCO downloads use the official `images.cocodataset.org` S3 bucket over HTTPS; VQA uses the `cvmlp` bucket linked by its official download page. No unofficial dataset replacement is used.
