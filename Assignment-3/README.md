# ATML PA3 Task 1 implementation

This repository implements the seven required evaluation conditions from the five Task 1 training runs. The official assignment PDF is `ATML-PA3.pdf`. No experimental scores are included; results are written only by actual Kaggle runs. The PDF report and scientific interpretation remain the student's work.

## Kaggle setup

1. Upload the current **audited** `Assignment-3` folder as a Kaggle dataset, including `first_plot_questions.yaml`, and attach it to a notebook. Enable **GPU** and **Internet**. The notebook can also sparse clone the public GitHub repository, but its embedded source hashes reject an older checkout; no changes have been pushed here.
2. Open `kaggle/ATML_PA3_Task1.ipynb`. Its one editable code cell finds the repository root, installs dependencies, restores any mounted output dataset, writes a runtime config, and runs the selected phase. Set `REPO_SOURCE` only if discovery cannot find your mount.
3. Run `diagnose`, `prepare`, and **`preflight`**, in that order. Read `outputs/task1/preflight/report.json`. Training requires a matching `data/preflight_pass.json` from the full GPU preflight. A `--cpu-only` preflight records GPU integration as `NOT RUN` and does not unlock training.
4. Set one `PHASE` per invocation. Every generated artifact goes under `/kaggle/working/outputs/task1` or `/kaggle/working/results/task1`; the Hugging Face cache defaults to `/kaggle/working/hf_cache`. Keep protocol settings fixed.

`metamath_dataset`, `gsm8k_dataset`, and `hellaswag_dataset` may point to Hugging Face IDs or local `save_to_disk` datasets attached under `/kaggle/input`. A local model directory may be used for `model`. Harness benchmark tasks still use their stock dataset IDs, so set `cache_dir` to a prepared Hugging Face cache if Internet is unavailable. The script sets `HF_HOME` and `HF_DATASETS_CACHE` from that path.

Python commands below work from the repository root after installation. The notebook calls the same modules.

```bash
python -m task1.run diagnose
python -m task1.run prepare
python -m task1.run preflight
python -m task1.run evaluate --condition baseline
python -m task1.run train --method full
python -m task1.run train --method l2sp
python -m task1.run train --method lora4
python -m task1.run train --method lora16
python -m task1.run train --method oplora16
python -m task1.run evaluate --condition full_mid
python -m task1.run evaluate --condition full_final
python -m task1.run evaluate --condition l2sp_final
python -m task1.run evaluate --condition lora4_final
python -m task1.run evaluate --condition lora16_final
python -m task1.run evaluate --condition oplora16_final
python -m task1.run weight
python -m task1.run activations
python -m task1.run qualitative --condition baseline
python -m task1.run qualitative --condition full_mid
python -m task1.run qualitative --condition full_final
python -m task1.run qualitative --condition l2sp_final
python -m task1.run qualitative --condition lora4_final
python -m task1.run qualitative --condition lora16_final
python -m task1.run qualitative --condition oplora16_final
python -m task1.run aggregate
```

For a local path override, pass `--config path/to/task1.yaml`. The notebook writes path overrides to `/kaggle/working/task1_runtime.yaml`. `task1/protocol.py` rejects changes to the assignment's experimental settings.

## Resume and export

Recovery checkpoints are saved at updates 312, 625, 937, and 1250, in separate method folders. Repeating a `train` command loads `latest.pt`, checks the config and selected dataset hash, restores optimizer, scheduler, scaler, RNG, and data position, then resumes at the next update. The training log is trimmed to the recovered step to avoid duplicate records. A skipped FP16 optimizer update retries the same 32 examples at a lower scaler value and is counted separately.

`inference_XXXX.pt` contains only the model weights for full methods or adapter state for adapter methods. OPLoRA also saves `projected_XXXX.pt`, containing effective projected factors. `recovery_XXXX.pt` and `latest.pt` contain complete state and can be large.

Full FT recovery files include FP32 model weights plus FP32 Adam moments. Four such files may exceed a standard Kaggle working directory. The trainer estimates disk needs before starting. Run `python -m task1.run train --method full --stop-update 312`, save that session, then resume with stop points 625, 937, and 1250. The scheduler still spans 1,250 updates. Restore only the latest recovery file and the needed inference files if older recovery files are already preserved in a prior Kaggle dataset. The same stop points are available for every method.

The harness computes complete-split primary metrics and logs every sample. Its optional bootstrap standard-error calculation is disabled (`bootstrap_iters=0`) because it does not affect the required accuracy scores and adds substantial CPU work.

Before a Kaggle session ends, use **Save Version** so Kaggle preserves `/kaggle/working`, then make those outputs available as a dataset for the next session. Attach the latest saved dataset under **Add Input** and set `RESTORE_DATASETS` in the one controller cell. It restores selected data, the preflight report and marker, results, and the chosen method's `latest.pt` (or newest recovery), logs, and inference files. Set `METHOD` to the method you want to resume; use a saved dataset containing every method's inference files for final diagnostics. If two mounted datasets contain different files at the same path, restoration stops rather than silently overwriting one. `python -m task1.run export --method METHOD` creates a ZIP of one method's full output folder after checking free disk space. Save results separately as well.

## Reproducibility and checks

Preparation selects exactly 40,000 examples in original MetaMathQA order. It records source indices, IDs, dataset fingerprints, rejection counts, lengths, and SHA256 of selected token sequences. GSM8K test questions are normalized for contamination checks. The Qwen3 chat template is called with `enable_thinking=False`. Only assistant tokens and the end-of-turn token have nonmasked labels. Each effective batch loss is the sum over valid response tokens divided by the total valid response-token count across all 32 examples.

The qualitative loader validates 24 prompts for eight underlying questions and three formats. It records the YAML SHA256 and rejects partial outputs or cross-condition files with changed prompt text, system instructions, or file fingerprint. Aggregation requires all 168 actual responses before producing the 42-row blank annotation template.

Full FT uses FP32 model parameters with FP16 autocast; adapter methods use a frozen FP16 backbone and FP32 factors. L2-SP keeps FP32 reference weights on pinned CPU memory and streams them to the GPU for the exact penalty and gradient. This saves GPU memory but adds transfer time. OPLoRA computes FP32 SVD bases once from the original FP32 weights, then trains projected factors without dense projection matrices. This preprocessing can be slow. If Kaggle lacks memory or disk for the prescribed experiment, stop and report the limitation; do not change the protocol.

Run fast tests with `python -m unittest discover -s tests -v`. A CPU-only preflight is available as `python -m task1.run preflight --cpu-only --config LOCAL_PATH_CONFIG`, where output and cache paths point inside a writable local directory. These checks do not verify the real tokenizer, datasets, or GPU. The full Kaggle preflight checks those interfaces and writes a pass marker only if all checks succeed. Full training and benchmark suites are not run locally.

The implementation uses the [OPLoRA paper](https://ojs.aaai.org/index.php/AAAI/article/view/40703) for the double-sided projection construction and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for stock task prompts, stopping, filters, and scoring. The implementation is original; no external source code is copied into this repository.

See [Task 1 traceability](docs/task1_traceability.md) for each required output and the current verification boundary.
See [Task 1 audit](docs/task1_audit.md) for defects, test status, and remaining Kaggle risks.
