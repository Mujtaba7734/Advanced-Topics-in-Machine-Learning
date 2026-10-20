# Assignment 2: LLM Post-Training

> **Authorship and AI-use disclosure:** I wrote the report myself. AI helped me generate its initial structure/scaffold, but I wrote the report text and analysis. I also used AI for coding assistance and suggestions; I wrote, adapted, and checked the final code myself.

This folder contains my Advanced Topics in Machine Learning Programming Assignment 2 work. It covers direct preference optimization (DPO), proximal policy optimization (PPO), group relative policy optimization (GRPO), safety evaluation, and a comparison of reinforcement learning with verifiable rewards (RLVR) and AI feedback (RLAIF). The assignment instructions are in [`manual/ATML PA2.pdf`](manual/ATML%20PA2.pdf).

## Repository layout

| Path | Contents |
| --- | --- |
| `task1_dpo/` | DPO training, evaluation, beta ablations, and length analysis |
| `task2_ppo/` | PPO continuation, evaluation, clipping analysis, and KL ablations |
| `task3_grpo/` | GRPO continuation, evaluation, group-size analysis, and normalization comparison |
| `task4_safety/` | Response generation, AI judging, safety metrics, and manual-audit preparation |
| `task5_feedback/` | Frozen-policy math evaluation, perturbation scoring, and RLVR/RLAIF comparison |
| `common/`, `configs/`, `scripts/` | Shared code, experiment settings, and asset/environment utilities |
| `Report/` | LaTeX report source, tables, figures, and evidence audit |
| `backups/` | Saved experiment-output archives and checksums |

## Setup

Run commands from the `Assignment-2` directory. From a fresh clone of this repository:

```bash
git clone https://github.com/Mujtaba7734/Advanced-Topics-in-Machine-Learning.git
cd Advanced-Topics-in-Machine-Learning/Assignment-2
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
python -m scripts.check_environment
```

The asset script downloads the fixed datasets, cached diagnostics, and supplied checkpoints from [`AbDu11aHHH/ATML-PA2-assets`](https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets) at revision `0b350481fb03f5525a35bcdec4131bd4fe487f98`. Public base, reward, and judge models may be downloaded separately when an experiment runs. The asset and model downloads can be large.

## Experiment entry points

The following commands run the task pipelines using the checked-in configurations. Tasks 1–4 include training or model inference and require suitable compute; Task 5 evaluates supplied frozen policies.

```bash
python -m task1_dpo.run_all --config configs/dpo.yaml
python -m task2_ppo.run_all --config configs/ppo.yaml
python -m task3_grpo.run_all --config configs/grpo.yaml --resume
python -m task4_safety.run_all --config configs/feedback.yaml --resume
python -m task5_feedback.run_all --config configs/feedback.yaml
```

Each task also has individual scripts for its training, evaluation, and analysis stages. Task 4 prepares a blind manual-audit sheet; its human labels require manual review and must not be filled from the AI judge's labels. Task 5 does not retrain the supplied RLVR or RLAIF policies.

## Report and reproducibility

The report source is [`Report/main.tex`](Report/main.tex); the underlying figures and tables are in `Report/figures/` and `Report/tables/`. [`Report/EVIDENCE_AUDIT.md`](Report/EVIDENCE_AUDIT.md) records where the saved evidence came from and notes limitations that matter when interpreting the results.

Experiment settings live in `configs/`. Keep the fixed data, prompt IDs, seeds, checkpoints, decoding settings, and evaluation procedures matched when reproducing comparisons. The saved backups and report data provide experiment records; rerunning the full pipelines requires the course assets and substantial compute.
