# Task 3: Test-Time Compute

Open `../colab/ATML_PA3_Task3_RunAll.ipynb`, select **one T4 GPU**, then **Runtime → Run all** and approve the Google Drive mount. Everything is isolated under `/content/drive/MyDrive/ATML_PA3_Task3/`. Run All again after a safe stop or disconnect. No manual data uploads, model changes, per-stage edits, or repeated generation of verified calls are needed.

The authoritative specification is `../ATML-PA3.pdf`, Section 3, with the official release README reproduced in `OFFICIAL_README.MD`. The official release was discovered from the PDF's hyperlink, not guessed. `release_manifest.json` pins the actual file IDs, byte sizes, and SHA-256 values, including the linked MathVista question metadata. A changed release fails explicitly. No images are downloaded. No report prose or experimental interpretation is generated.

## Implementation and evidence

| Requirement | Implementation | Evidence under the versioned run directory |
|---|---|---|
| Official joins and token alignment | `data.py`, `audit.py` | release manifest, schemas, exact split IDs; row and token validation |
| Nested 10-order sampling, N=1…32 | `core.py`, `offline.py` | `orders.json`, selections CSV/Parquet, question means |
| All mandated selectors, oracle, miss decomposition | `core.py`, `offline.py` | summary, candidate IDs, consumed IDs, selected answer/rollout, tokens |
| Length/signal associations and fusion ablation | `offline.py`, `export.py` | per-question quartiles/correlations, token associations, features and plots |
| Live 50-question, 3-strategy comparison | `live.py` | all 250 call records, 150 strategy records, timings, tokens, memory, transitions |
| Dev-only adaptive fitting | `export.fit_policies`, `core.gate` | full dev grid, frozen policies, held-out stopping counts and cost comparisons |
| Paired 1,000-bootstrap uncertainty | `core.paired_bootstrap` | question-level paired intervals, including live and adaptive comparisons |
| Examples without outcome selection | `offline.py`, `export.py` | first three saved evaluation/live IDs, complete candidates, fixed trajectories, and deterministic retrospective case categories |
| Recovery and completion | `storage.py`, `run.py` | immutable checked objects, latest/previous pointers, identity manifest, final verification |

The offline token costs are **proxies**, not GPU runtime. Confidence/attention analyses are **associations**, not causal grounding evidence. The optional extra mean-logp RWC selector is retained alongside all required selectors. The shortest-rollout control uses only length.

## Frozen choices where the assignment leaves details open

- Order and live-call seeds are SHA-256-derived from seed 6304, dataset/question, order/call name. They are independent of execution order and stable after interruption. The exact split uses lexicographic string IDs followed by `default_rng(6304)` exactly as specified.
- Scorer: `math-verify==0.8.0`. Bare released MATH answers are wrapped in a box for the package's parser. Exact strings compare directly; other expressions use math-verify. Grouping uses revealed candidates in order, matching the first equivalent representative; gold never enters grouping. This also specifies behavior if a symbolic equivalence relation is imperfect/nontransitive.
- MathVista follows the linked project's `evaluation/calculate_score.py` normalization: option letters/parenthesized letters; otherwise closest choice by Levenshtein distance with first-choice ties; integer conversion with `int(float(x))`; float rounding with Python `round(float(x), precision)`; lists as strings. Choices are returned as letters as the assignment requires. Missing/invalid numeric extractions become one `NO_ANSWER` group and are always incorrect. The source URL and retrieved source hash are recorded in `verification.json`. Stored extracted answers are used, without a new LLM extractor.
- Quartiles contain exactly eight candidates. Sorting uses score followed by canonical rollout-ID order; lowest/highest eight form bottom/top. Correlations are Pearson, undefined constant-array correlations are null and their coverage is retained in raw files. Average ranks handle RWC ties, while vote and max-score ties use earliest support in the current permutation.
- Invalid/empty signals receive negative infinity for ranking; all-invalid score ties remain in the denominator and resolve by earliest support. Undefined correlations are null, never invented zeros. Nonfinite or misaligned official source arrays fail validation. Zero eligible attention totals are treated as invalid visual signals.
- FirstAns consumes one candidate at every N. Oracle/distinct-group opportunity and miss decomposition are recorded against the common revealed N-candidate opportunity pool; consumed IDs and actual FirstAns cost are recorded separately.
- Each adaptive family includes always-32 (`tau=2`) as a fallback. Only development records enter threshold selection. Policies are durably frozen before any held-out evaluation. Retrospective full-pool correctness is added solely to output records and never passed into selectors or gates.
- Illustrative token trajectories use rollout 0 from the first three evaluation IDs; all 32 candidate trajectories/texts are saved for those questions. This trajectory selection is independent of correctness, scores, or which method wins. Separate case records select the first up to three qualifying questions in the fixed evaluation order for answer disagreement, incorrect early stops, and fixed-32-recoverable failures. Live examples additionally include first-in-order wrong-to-correct and correct-to-wrong revisions. These category labels use gold retrospectively only; empty categories are explicitly retained.

## Durable caching and time budgets

Downloads, pinned Hugging Face model snapshots, logs, and all experiment state live on Drive. Verified Parquet files are copied to local SSD for analysis. Only a question's token arrays are loaded at a time. Question checkpoints store features, selections, and adaptive results; completed live calls are committed individually before the next call.

Run identity includes the source commit, source hashes, protocol/seed, exact data hashes, model revision, Python, and relevant dependency versions. Different identities get separate run folders. Compatible preinstalled CUDA PyTorch is preserved. Exact pinned experiment dependencies are installed only if missing/incompatible; fresh child processes avoid stale notebook imports.

Objects are immutable and checksum-verified. Pointer updates are atomic with readback; verified previous pointers are retained. Partial writes cannot mark a task complete. A generation interrupted mid-call restarts that one call with its original seed; completed calls are reused. No Task 1 or Task 2 paths are modified or cleaned.

The default session budget is five hours with a 15-minute reserve. This is a **safe-stop budget, not an estimated completion time**. Offline checks occur between questions; live checks occur during generation as well as between calls. Unfinished calls are never accepted as completed. A real T4 session first runs separate forced-length 3,072-token and long-prefill 2,048-token memory diagnostics. These diagnostic outputs are excluded from experimental metrics, and their memory/time records are saved separately. Actual experimental calls retain the required early-stop behavior and caps.

Logs appear continuously in the notebook and are saved as `full_pipeline.log` and `progress.jsonl`. A 30-second heartbeat covers long downloads/CPU/model-loading phases; live token progress appears approximately every 20 seconds. `status.json`, `heartbeat.json`, `latest_run.json`, and `final_verification.json` distinguish failure, safe stop, partial completion, and full completion. Run All performs unit tests before experiments. A return code of zero is insufficient: the final cell independently rehashes required artifacts and requires two completed offline datasets, 50 live questions, 250 live calls, and four policies.

## Tests and local reproduction

From `Assignment-3`:

```bash
python -m pytest -q tests/test_task3.py
python -m task3.audit --official-root /path/to/official --output audit.json
python -m task3.integration --official-root /path/to/official --metadata /path/to/testmini.json --output /path/to/test-evidence
python -u -m task3.run --root /path/to/durable/task3
```

See `verification.json` for checks actually executed during implementation. CPU tests and official-data validation do not establish T4 compatibility, full-experiment completion, or runtime. No full offline benchmark or live experiment is claimed to have run during implementation. The Colab notebook must complete those experiments and produce a truthful final verification record.

## Sources

- Course PDF and official [rollout release](https://drive.google.com/drive/folders/1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23).
- [Math-Verify](https://github.com/huggingface/Math-Verify), used as a dependency.
- [MathVista normalization](https://github.com/lupantech/MathVista/blob/main/evaluation/calculate_score.py) and [testmini metadata](https://github.com/lupantech/MathVista/blob/main/data/testmini.json), linked from the course PDF. The small normalizer is independently implemented to follow those published rules.
- [Pinned Qwen model](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/tree/aa8e72537993ba99e69dfaafa59ed015b17504d1).
