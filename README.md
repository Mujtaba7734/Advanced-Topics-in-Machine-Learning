# Advanced Topics in Machine Learning

Course assignments and reproducible experiment implementations.

- [PA3 Task 3: test-time compute](Assignment-3/task3/README.md)
- [PA3 Task 3 single-T4 Colab Run All notebook](Assignment-3/colab/ATML_PA3_Task3_RunAll.ipynb)
- [PA3 Task 3 dual-T4 Kaggle Run All notebook](Assignment-3/kaggle/ATML_PA3_Task3_Kaggle_RunAll.ipynb)

Task 3 is isolated from the existing Task 1 and Task 2 implementations and results. Its README documents the required data, fixed protocol, evidence outputs, recovery, and actual verification boundary.

The Kaggle launcher assigns entire live questions to two identical T4s while retaining FP16, batch size 1, and the original prompts, seeds, token budgets, and per-call timing. Each question stays on its assigned GPU slot when resumed. The offline stage runs on CPU. The launcher uses one shared 6.18 GB model cache and the 0.42 GB official release. A local size audit of eight development and eight evaluation questions per dataset projects about 1.12 GB of offline checkpoints and about 0.2 GB of offline exports. The full run is estimated at 8–9 GB; the launcher checks a 12 GB budget plus 1.5 GB free space and rechecks disk space during execution. A real Kaggle GPU run is still needed to measure elapsed time and confirm peak disk and GPU memory use.
