# ATML PA3 Task 1 implementation

This repository implements the seven required evaluation conditions from the five Task 1 training runs. The official assignment PDF is `ATML-PA3.pdf`. No experimental scores are included; results are written only by actual Kaggle runs. The PDF report and scientific interpretation remain the student's work.

## Kaggle setup

1. Upload the current **audited** `Assignment-3` folder as a Kaggle dataset, including `ATML-PA3.pdf`, `first_plot_questions.yaml`, and `kaggle/source_manifest.json`. Attach it to the Kaggle notebook. Enable **T4 ×2** and **Internet**. The notebook can sparse clone the public GitHub repository only when that checkout exactly matches its embedded source manifest; no changes have been pushed here.
2. Import [ATML_PA3_Task1_RunAll.ipynb](kaggle/ATML_PA3_Task1_RunAll.ipynb) and click **Run All**. The 18 separate cells select their own stage and verify dependencies in a fresh Python process. Nothing needs to be edited between stages. All outputs are under `/kaggle/working/outputs/task1` and `/kaggle/working/results/task1`; the Hugging Face cache is `/kaggle/working/hf_cache`.
3. If training stops at a required checkpoint, use Kaggle **Save Version** to preserve `/kaggle/working`, attach that saved output as an input dataset in a new session, and click **Run All** again. The restore stage verifies the mounted recovery file before resuming. If an interrupted run has not reached an official checkpoint, the current session must be recovered before ending or that unsaved work is lost.
4. Review `/kaggle/working/results/task1/summary/final_verification.json` and `/kaggle/working/results/task1/summary/backup_manifest.json`. The compact ZIP does not contain full checkpoints; Kaggle Save Version is required for those. A CPU-only preflight never unlocks training.

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

For a local path override, pass `--config path/to/task1.yaml`. The Run All notebook writes path overrides to `/kaggle/working/task1_runtime.yaml`. `task1/protocol.py` rejects changes to the assignment's experimental settings. Regenerate the source manifest and notebook together with `python kaggle/build_notebook.py` after source changes.

## Resume and export

Recovery checkpoints are saved at updates 312, 625, 937, and 1250, in separate method folders. Repeating a `train` command loads `latest.pt`, checks the config and selected dataset hash, restores optimizer, scheduler, scaler, RNG, and data position, then resumes at the next update. The training log is trimmed to the recovered step to avoid duplicate records. A skipped FP16 optimizer update retries the same 32 examples at a lower scaler value and is counted separately.

`inference_XXXX.pt` contains only the model weights for full methods or adapter state for adapter methods. OPLoRA also saves `projected_XXXX.pt`, containing effective projected factors. `recovery_XXXX.pt` and `latest.pt` contain complete state and can be large.

On a two-GPU Kaggle session, Full FT and L2-SP keep their FP32 model and gradients on GPU 0, while ordinary PyTorch AdamW updates FP32 mirror weights and moments on GPU 1. The FP32 objective, AdamW groups, schedule, and batch are unchanged. LoRA methods use GPU 0; OPLoRA computes its FP32 SVD workspace on GPU 1. With only one GPU, Full FT and L2-SP use the original single-GPU AdamW and must pass the same preflight. The preflight now performs three consecutive complete 32-example optimizer updates per method using the longest prepared sequences, then checks recovery and at least 1 GiB peak reserved-memory headroom on each GPU. A prior successful preflight marker is invalidated by code changes or different GPU hardware.

Full FT recovery files include FP32 model weights plus FP32 Adam moments. The secondary-GPU mirror weights are reconstructed from the model checkpoint and are not saved twice. Four recovery files may exceed a standard Kaggle working directory. The Run All planner uses the preflight's measured checkpoint size, update time, current free disk, and a safety margin to choose the furthest reachable official checkpoint. The trainer independently checks disk and host RAM before checkpointing. The scheduler still spans 1,250 updates. Existing native AdamW recovery files can migrate to the secondary-GPU optimizer. Restore the latest recovery file and the needed inference files from a saved output dataset; the same stop points are available for every method.

The harness computes complete-split primary metrics and logs every sample. Its optional bootstrap standard-error calculation is disabled (`bootstrap_iters=0`) because it does not affect the required accuracy scores and adds substantial CPU work.

Before a Kaggle session ends, use **Save Version** so Kaggle preserves `/kaggle/working`, then attach those outputs under **Add Input** in the next session. The Run All restore stage scans mounted output datasets automatically. It preserves working files, rejects ambiguous mounted versions, and verifies the expected checkpoint hash before resuming. The compact export ZIP contains results and reproducibility files; its manifest lists full recovery and inference checkpoints separately and records that their durable backup is unconfirmed until Kaggle Save Version succeeds.

## Reproducibility and checks

Preparation selects exactly 40,000 examples in original MetaMathQA order. It records source indices, IDs, dataset fingerprints, rejection counts, lengths, and SHA256 of selected token sequences. GSM8K test questions are normalized for contamination checks. The Qwen3 chat template is called with `enable_thinking=False`. Only assistant tokens and the end-of-turn token have nonmasked labels. Each effective batch loss is the sum over valid response tokens divided by the total valid response-token count across all 32 examples.

The qualitative loader validates 24 prompts for eight underlying questions and three formats. It records the YAML SHA256 and rejects partial outputs or cross-condition files with changed prompt text, system instructions, or file fingerprint. Aggregation requires all 168 actual responses before producing the 42-row blank annotation template.

Full FT uses FP32 model parameters with FP16 autocast; adapter methods use a frozen FP16 backbone and FP32 factors. L2-SP keeps FP32 reference weights on pinned CPU memory and streams them to the GPU for the exact penalty and gradient. This saves GPU memory but adds transfer time. OPLoRA computes FP32 SVD bases once from the original FP32 weights, then trains projected factors without dense projection matrices. This preprocessing can be slow. If Kaggle lacks memory or disk for the prescribed experiment, stop and report the limitation; do not change the protocol.

Run fast tests with `python -m unittest discover -s tests -v`. A CPU-only preflight is available as `python -m task1.run preflight --cpu-only --config LOCAL_PATH_CONFIG`, where output and cache paths point inside a writable local directory. These checks do not verify the real tokenizer, datasets, or GPU. The full Kaggle preflight checks those interfaces and writes a pass marker only if all checks succeed. It reuses validated prepared data, so a new Kaggle session needs only a restored data and cache mount plus one full preflight run. Full training and benchmark suites are not run locally.

The implementation uses the [OPLoRA paper](https://ojs.aaai.org/index.php/AAAI/article/view/40703) for the double-sided projection construction and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for stock task prompts, stopping, filters, and scoring. The implementation is original; no external source code is copied into this repository.

See [Task 1 traceability](docs/task1_traceability.md) for each required output and the current verification boundary.
See [Run All audit](docs/task1_runall_audit.md) for local test status and remaining Kaggle verification, and [Task 1 audit](docs/task1_audit.md) for the training implementation.
