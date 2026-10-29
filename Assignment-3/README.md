# ATML PA3 Task 1 implementation

This repository implements the seven required evaluation conditions from the five Task 1 training runs. The official assignment PDF is `ATML-PA3.pdf`. No experimental scores are included; results are written only by actual Kaggle runs. The PDF report and scientific interpretation remain the student's work.

## Kaggle setup

1. Upload this repository as a Kaggle dataset named `atml-pa3`, or clone it to `/kaggle/working/ATML-PA3`. The assignment's `first_plot_questions.yaml` is included at the repository root. Enable **GPU** and **Internet** in Notebook Settings.
2. Add the repository dataset to a new notebook and open `kaggle/ATML_PA3_Task1.ipynb`. Edit `REPO` in its first code cell if Kaggle mounts the input under another name.
3. Run the installation cell. It installs the versions in `requirements.txt` and prints GPU VRAM and free disk space. Restart the kernel if Kaggle reports that an imported package was replaced.
4. Set one `PHASE` in the controller cell. Every generated artifact goes under `/kaggle/working/outputs/task1` or `/kaggle/working/results/task1`; Hugging Face cache defaults to `/kaggle/working/hf_cache`. Edit `configs/task1.yaml` to change paths only, keeping experiment settings fixed.

`metamath_dataset`, `gsm8k_dataset`, and `hellaswag_dataset` may point to Hugging Face IDs or local `save_to_disk` datasets attached under `/kaggle/input`. A local model directory may be used for `model`. Harness benchmark tasks still use their stock dataset IDs, so set `cache_dir` to a prepared Hugging Face cache if Internet is unavailable. The script sets `HF_HOME` and `HF_DATASETS_CACHE` from that path.

Python commands below work from the repository root after installation. The notebook calls the same modules.

```bash
python -m task1.run diagnose
python -m task1.run prepare
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

For a local path override, pass `--config path/to/task1.yaml`. No phase silently changes batch size, sequence length, precision, dataset size, benchmark split, or model.

## Resume and export

Recovery checkpoints are saved at updates 312, 625, 937, and 1250, in separate method folders. Repeating a `train` command loads `latest.pt`, checks the config and selected dataset hash, restores optimizer, scheduler, scaler, RNG, and data position, then resumes at the next update. The training log is trimmed to the recovered step to avoid duplicate records. A skipped FP16 optimizer update retries the same 32 examples at a lower scaler value and is counted separately.

`inference_XXXX.pt` contains only the model weights for full methods or adapter state for adapter methods. OPLoRA also saves `projected_XXXX.pt`, containing effective projected factors. `recovery_XXXX.pt` and `latest.pt` contain complete state and can be large.

Full FT recovery files include FP32 model weights plus FP32 Adam moments. Four such files may exceed a standard Kaggle working directory. The trainer estimates disk needs before starting. Run `python -m task1.run train --method full --stop-update 312`, save that session, then resume with stop points 625, 937, and 1250. The scheduler still spans 1,250 updates. Restore only the latest recovery file and the needed inference files if older recovery files are already preserved in a prior Kaggle dataset. The same stop points are available for every method.

The harness computes complete-split primary metrics and logs every sample. Its optional bootstrap standard-error calculation is disabled (`bootstrap_iters=0`) because it does not affect the required accuracy scores and adds substantial CPU work.

Before a Kaggle session ends, use **Save Version** so Kaggle preserves `/kaggle/working`, then make those outputs available as a dataset for the next session. Attach it under **Add Input**. In the notebook restore cell add its `/kaggle/input/...` mount to `RESTORE_DATASETS`. For training set `RESTORE_MODE='train'` and the method name; the cell restores its `latest.pt`, logs, inference files, selected data, and results. Older recovery files stay in the attached input dataset and do not consume working disk. For evaluation or diagnostics set `RESTORE_MODE='analysis'` and attach all relevant saved datasets; the cell copies inference files without recovery state. Then rerun the desired phase. `python -m task1.run export --method METHOD` creates a ZIP of one method's full output folder after checking free disk space. Save results separately as well.

## Reproducibility and checks

Preparation selects exactly 40,000 examples in original MetaMathQA order. It records source indices, IDs, dataset fingerprints, rejection counts, lengths, and SHA256 of selected token sequences. GSM8K test questions are normalized for contamination checks. The Qwen3 chat template is called with `enable_thinking=False`. Only assistant tokens and the end-of-turn token have nonmasked labels. Each effective batch loss is the sum over valid response tokens divided by the total valid response-token count across all 32 examples.

The qualitative loader validates 24 prompts for eight underlying questions and three formats. It records the YAML SHA256 and rejects partial outputs or cross-condition files with changed prompt text, system instructions, or file fingerprint. Aggregation requires all 168 actual responses before producing the 42-row blank annotation template.

Full FT uses FP32 model parameters with FP16 autocast; adapter methods use a frozen FP16 backbone and FP32 factors. L2-SP keeps FP32 reference weights on pinned CPU memory and streams them to the GPU for the exact penalty and gradient. This saves GPU memory but adds transfer time. OPLoRA computes FP32 SVD bases once from the original FP32 weights, then trains projected factors without dense projection matrices. This preprocessing can be slow. If Kaggle lacks memory or disk for the prescribed experiment, stop and report the limitation; do not change the protocol.

Run fast synthetic tests with `python -m unittest discover -s tests -v`. These validate local math and state behavior, not full-scale model execution. Full training and benchmark suites are intentionally never run locally.

The implementation uses the [OPLoRA paper](https://ojs.aaai.org/index.php/AAAI/article/view/40703) for the double-sided projection construction and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for stock task prompts, stopping, filters, and scoring. The implementation is original; no external source code is copied into this repository.

See [Task 1 traceability](docs/task1_traceability.md) for each required output and the current verification boundary.
