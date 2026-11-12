# Task 2 requirement traceability

Authoritative specification: `Assignment-3/ATML-PA3.pdf` pp. 6–10. This document maps implementation to evidence; the existence of a runtime check does **not** mean it was executed on CUDA. Actual development evidence is in `task2_validation.json`; full-run evidence is generated in the run directory on Drive. No report interpretation is authored here.

| Requirement | Implementation | Test / runtime evidence |
|---|---|---|
| Frozen original SigLIP, 196×768, spatial order, final norm once | `task2/features.py:Features.prepare`, pinned revision in `protocol.py` | `test_siglip_final_norm_once`; actual pretrained CPU fixture in validation; runtime hook, shape/dtype/finite gates |
| Deterministic preprocessing, FP16 cache, encoder removed before training | `features.py` | Actual CPU processor/cache round-trip; `feature_manifest.json`; allocator cleanup before Qwen load |
| Connector 768→1024→1024 GELU | `model.py:Connector` | Tiny and pretrained model gradient tests; `trainable_parameters.json` |
| Training-only magnitude calibration | `model.py:calibrate`, first 64 selected captions | Reject non-train split; `calibration.json` records RMS rule/scale |
| Original Qwen3-0.6B frozen, thinking disabled | `model.py:load_model`, `VLM.__init__`, `VLM.audit` | Actual tokenizer check; `test_lora_trainability_and_gradients`; runtime dtype/trainability audit |
| No quantization, FP16 autocast, FP32 trainables/AdamW | `model.py:stage`, `train.py:update`, `preflight.py` | Runtime parameter and optimizer-state dtype gates; CUDA remains unverified until runtime |
| Gradient checkpointing with connector path through frozen LM | `model.py:losses`, non-reentrant checkpointing | `test_loss_matches_full_logits_and_gradient`, pretrained CPU gradient fixture |
| Visual tokens before question inside native user message | `model.py:pack` | `test_pack_masks_and_no_image`; actual Qwen template inspected |
| Label mask: visual/prompt/pad ignored, answer+EOT supervised | `model.py:pack` | `test_pack_masks_and_no_image` |
| Per-example token mean, then effective-batch mean | `model.py:losses`, `train.py:update` | Full-logit equality and accumulated-vs-batched gradient tests |
| Exact CE without visual-prefix vocabulary logits | Answer-position projection + chunked head checkpointing in `model.py` | `test_loss_matches_full_logits_and_gradient`; GPU rates measured at runtime |
| Stage 1 two complete passes, 2250 updates, LR 1e-3 | `protocol.py:schedule`, `train.py`, `run.py` | `test_schedule`; `sampling_plan.json`, update IDs/logs, final verifier |
| Stage 2 1500 updates, pair-draw .5/.25/.25, LR 1e-4/2e-5 | `protocol.py`, `train.py` | Seed replay/pair preservation/distribution tests; realized mixture counts |
| Stage 3 750 updates, .7/.1/.1/.1, LR 5e-5/1e-5 | `protocol.py`, `train.py` | Same sampler tests; per-update mixtures and stage boundaries |
| LoRA q/k/v/o only, r16, alpha32, dropout .05, no biases | `model.py:stage` | Trainability and gradient tests; runtime parameter counts |
| Seed 6304; microbatch8 × accumulation2; 4500/72000 | `protocol.py`, `train.py`, `run.py:verify_final` | `test_schedule`, `test_accumulation_example_weighting`; runtime exact counters |
| AdamW .01, clip1 after unscale, per-stage cosine/.1 floor | `train.py`, `protocol.py:lr_factor` | `test_lr`; recorded learning rates/norms/overflow retries |
| Question80/caption64/short16 caps | `model.py:tokenize` | Cached tokenized examples; maximum-length real-CUDA stress fixtures |
| 6000 train images×3 captions, dev96×1 | `data.py:prepare/validate` | Real official-data selection verified in development; manifest counts/hashes |
| Presence and count1–4 annotation-derived pairs | `data.py:grounding` | Original category IDs/counts saved; pair pools; documented crowd policy |
| Grounding dev24/final64 pairs | `data.py:prepare/validate` | Real selection counts; balanced subtype allocation is predeclared |
| VQA1536/32/96; type640/256/640,12/8/12,36/24/36 | `data.py:select_natural/validate` | Real official-data selection checks; `test_natural_real_schema_and_consensus_filter` |
| VQA different normalized answers; >=8/10 both; original target/all refs | `data.py:natural_candidates`, `metrics.py` | Real selection validation; consensus fixture; raw ten-answer manifests |
| Train2014/val2014 isolation, all eval suites image-disjoint | `data.py:validate` | Real selection set intersections; replay split validation |
| Selected image/question/pair IDs and original source indices | `data.py` | `selected_data.json`, source-archive SHA256 records, plan hash |
| Development before train, every500, endpoints2250/3750/4500 | `run.py`, `evaluate.py:development` | Exact required evaluation set in `verify_final`; dev JSON at runtime |
| Dev caption loss, VQA NLL correct/swap/no-image | `evaluate.py:development` | NLL mask test; raw dev examples and aggregate swap penalty |
| Final greedy16, batch8, fixed thinking-disabled prompt | `model.py:generate_answers`, `evaluate.py` | `test_generation_passes_dynamic_positions`; pretrained CPU inference fixture |
| Four image controls, original references fixed | `evaluate.py:controlled` | `test_controls_keep_original_references`; fixed mismatched mapping saved before generation |
| No image actually omits all visual tokens/rebuilds positions | `model.py:pack` | Exact196 length-difference test; actual generation gate in CUDA preflight |
| Grounding normalized EM, VQA local soft consensus | `metrics.py` | Hand-checked 0/.3/.6/.9/1 agreement fixtures; normalization tests |
| Per-type scores, pair-both-correct, identical/empty rates | `metrics.py:pair_vectors/summarize` | Complete-pair validation and pair-accuracy fixtures; raw final files |
| Reference NLL excludes EOT, averages per example | `model.py:losses(include_eot=False)` | Mask fixture and full-logit equivalence test |
| Generation seconds/answer, swap advantages | `evaluate.py`, `metrics.py` | Synchronized CUDA timers; NLL sign/accuracy difference fixture |
| 4000 complete-pair bootstrap, paired conditions | `metrics.py:bootstrap/summarize` | Hand-checked constant/difference CI tests; deterministic resampling |
| Follow partner answer vs unsupported changes | `evaluate.py:final` | Per-example `follows_partner`, per-condition summary |
| Learning curves and marked stage boundaries | `evaluate.py:plots` | `learning_curves.png` from actual update/dev logs only |
| Predeclared qualitative selection and paired images | `evaluate.py:qualitative_html` | First4 lexical pair IDs per type; raw JSON and aligned self-contained HTML |
| Timing, trainable counts, memory separate cache/train/eval | `features.py`, `preflight.py`, `run.py` | `timing.json`, `memory.json`, per-stage counts, preflight timings |
| Optional49 tokens off by default | Mandatory-only `P.visual_tokens=196`; no compression launcher | No shortened-run substitution; optional experiment intentionally absent |
| Single thin Run All notebook, precise source commit | `colab/ATML_PA3_Task2_RunAll.ipynb`, `bootstrap.py` | Notebook syntax/schema validation; detached immutable checkout |
| Compatible packages reused, fresh subprocess | `bootstrap.py`, notebook | Version checks + pip constraints; no blind Torch reinstall |
| Normal authorized Drive mount, stable run identity | Notebook, `run.py` | Source/protocol/model/data/plan/version identity; no credentials stored |
| Local hot features/tokenization/logs; durable reusable assets | `features.py`, `run.py`, `storage.py` | Per-shard keys/checksums, copy timing, cached tokenization; checkpoint log history |
| 250-update recovery + endpoints + safe stop | `run.py`, `storage.py:Recovery` | Snapshot payload and verifier; previous snapshot retained; heartbeat backup cursor |
| Exact weights/optimizer/scheduler/scaler/RNG/sampler/logs | `train.py:payload`, `Recovery` | CPU Adam/RNG resume equality; CUDA interrupted-vs-continuous gate |
| Interrupted copy cannot become latest | `storage.py:publish/Recovery.save` | `test_recovery_and_interrupted_copy`, corrupted-latest fallback test |
| Safe session stop with 15-minute reserve | `run.py` deadlines, feature/evaluation batch checks | Safe-stop state; resumes without modifying protocol |
| GPU 8×2 worst-length stress; isolated16×1 benchmark | `preflight.py:smoke_stage` | Three consecutive complete8×2 updates + replay equivalence; actual GPU results required |
| No unapproved4×4/32×1, BF16, quantization or token reduction | Fixed protocol, strict update validation | Runtime hard gates; no fallback code paths |
| Honest ETA and five-hour claims | `run.py` projections | Measured stage/eval/copy extrapolations, unknown-cost notes; five-hour verified flag remains false |
| Genuine completion only | `run.py:verify_final` | Checks 4500 distinct updates,72000 presentations,12 dev points,8 complete final conditions, checksummed outputs |
| Task 1 unchanged; new branch only | All new paths scoped to Task 2 | Git diff against implementation parent; no main merge |
| No AI-authored report prose | Source/runtime evidence only | No report file created |

External prerequisites are deliberately hard failures: CUDA/T4 allocation, normal Drive consent, sufficient disk/RAM/VRAM, official endpoint availability, and model download availability. The user need not copy cells or edit paths. A hardware allocation or authorization failure cannot be repaired by silently changing the experiment.
