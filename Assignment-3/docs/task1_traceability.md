# Task 1 requirements traceability

The assignment PDF is authoritative. Paths below are relative to the configurable `output_root` and `result_root` in `configs/task1.yaml`. This checklist maps implementation and expected outputs; it does not claim that a Kaggle run has completed.

| PDF requirement | Implementation | Actual-run output |
|---|---|---|
| Original Qwen3-0.6B, seed 6304, independent starts | `task1/model.py`, `common/seed.py`, `task1/training.py` | each method's `config.json`, recovery files |
| 40,000 eligible MetaMathQA examples in order, GSM8K leakage filter | `task1/prepare_data.py` | `data/selected.json`, `data/manifest.json` |
| Non-thinking chat template and response-only masking | `task1/prepare_data.py`, `task1/dataset.py` | selected token and label sequences |
| One epoch, batch 2 x accumulation 16, 1,250 updates, token-normalized SFT | `task1/training.py`, `configs/task1.yaml` | `METHOD/train.jsonl` |
| FP16 computation, FP32 master or adapters, dynamic scaling, clipping, scheduler, AdamW exclusions | `task1/training.py`, `task1/model.py` | training log and recovery state |
| Full FT, midpoint and final displacement | `task1/training.py`, `task1/l2sp.py` | `full/train.jsonl`, `full/inference_0625.pt`, `full/inference_1250.pt` |
| L2-SP over all unique parameters, CPU reference and transfer overhead | `task1/l2sp.py`, `task1/training.py` | `l2sp/train.jsonl`, `l2sp/inference_1250.pt` |
| LoRA ranks 4 and 16, seven targets per layer, independent runs | `task1/model.py`, `task1/training.py` | `lora4/`, `lora16/` inference and recovery files |
| OPLoRA rank 16, protection rank 128, projected training and export | `task1/model.py`, `task1/training.py` | `oplora16/projected_1250.pt`, `export_1250.json` |
| Recovery at 312, 625, 937, 1250 with complete state | `common/checkpoint.py`, `task1/training.py` | `METHOD/recovery_XXXX.pt`, `latest.pt` |
| Seven conditions, four complete zero-shot benchmarks and per-example outputs | `task1/evaluation.py` | `evaluation/CONDITION/TASK.json`, `aggregate.json` |
| Baseline-relative percentage point changes and benchmark table | `task1/plots.py` | `summary/benchmarks.csv`, `benchmarks_wide.csv`, `benchmarks.png` |
| SFT curves, L2-SP diagnostics, resources | `task1/plots.py` | `summary/sft_loss.png`, `l2sp.png`, `resources.csv` |
| Weight diagnostics on down_proj layers 0, 13, 27 | `task1/diagnostics.py` | `diagnostics/weight_space.csv`, `summary/weight_space.png` |
| Fixed 32+32 probe IDs and original-model activation cache | `task1/diagnostics.py` | `data/probe_ids.json`, `data/baseline_activations.pt` |
| Math and non-math activation ratios | `task1/diagnostics.py` | `diagnostics/activation_ratios.csv`, `summary/activation_ratios.png` |
| 24 prompts x seven conditions; 42-response annotation subset | `task1/qualitative.py`, `first_plot_questions.yaml` | `qualitative/prompts_manifest.json`, `CONDITION.json`, `all_168_responses.csv`, `annotation_template.csv` |
| Kaggle phase controller, resume, export | `kaggle/ATML_PA3_Task1.ipynb`, `task1/run.py` | configurable working directories, method ZIPs |
| Synthetic correctness tests | `tests/test_task1.py` | console test report |

## Current blockers and verification boundary

- The assignment's `first_plot_questions.yaml` is present and its 24 prompts, eight question IDs, three formats, and eight JSON system instructions pass structural validation. Generation still requires the seven model conditions to be trained or loaded on Kaggle.
- This development environment has CPU-only PyTorch and no Hugging Face dataset cache. Full model, dataset, SVD, training, and benchmark integration have not been exercised here. The pinned dependency installation also requires Kaggle Internet or a prepopulated package cache.
- The PDF's research questions and required PDF report call for student interpretation. This repository exports data, plots, and a blank annotation template only.
