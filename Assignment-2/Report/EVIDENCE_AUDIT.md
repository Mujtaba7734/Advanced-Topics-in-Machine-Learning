# Engineering evidence audit (not report prose)

**Scope:** Tasks 1–5 quantitative records in the five saved backup archives, plus aggregate Task 4 manual-audit counts from the completed local workbook. Manual labels still require independent human validation.

## Present experiment runs

- Task 1: Standard one-epoch DPO (1500 pairs, 94 optimizer steps); beta short forks 0.03/0.10/0.30; length-balanced model, fixed length strata and word-limit prompts. Standard DPO is NOT matched in training budget with the short beta forks.
- Task 2: Standard PPO midpoint continuation 20/20; cached clip epsilon 0.05/0.20/0.50; matched eight-update clip short forks; eight-update KL forks beta 0/0.1/0.2. Critic loss/gradient, entropy, policy gradient and length are in standard_train.jsonl.
- Task 3: Standard GRPO midpoint continuation 20/20; cached K=2/4/8 equal-total-generation diagnostics (192 total); eight-update normalization GRPO/Dr.-GRPO forks, shared held-out evaluation. Four standard updates have zero optimization tokens after truncated completions are masked. This is not equivalent to 20 nonzero policy updates.
- Task 4: 450 fixed XSTest prompts per frozen policy; 1800 model responses and judge rows. The completed 60-prompt workbook contains 240 human labels; pooled agreement is 167/240, with confusion counts in `data/15_manual_audit.json`. The course judge assigned zero OVER_REFUSAL across ALL policies and mislabels some benign prompts. The automated judge MUST NOT be represented as human ground truth.
- Task 5: Fixed 300 GSM8K and 100 SVAMP examples across SFT/RLVR/RLAIF; 100 supplied controlled responses (20 problems x five variants), 80 perturbation pairs; course judge outputs and comparisons are saved. The course-supplied RLVR/RLAIF checkpoints are not trained as part of this assignment.
- Task 6: no new training required.

## Scientific cautions that must be discussed, not hidden

1. PPO epsilon short forks have exactly identical held-out metrics for all three epsilon values; the recorded on-policy standard clip fraction is 0 at all 20 updates. The *separate cached-rollout* affected-token rates differ substantially. Distinguish on-policy update geometry from cached probes. Do not invent a monotonic epsilon effect.
2. GRPO has four standard updates with no optimizable tokens from truncation masking; one additional update has zero group reward SD. Do not interpret flat loss as strong optimization without checking mask and gradient.
3. DPO standard one-epoch run vs beta short forks use unequal budgets. Describe them as separate controlled comparisons.
4. The AI safety judge's zero over-refusal rate is unreliable because the classifier sometimes calls benign answers JUSTIFIED_REFUSAL. Judge calibration is itself a limitation.
5. Task 5 AI pairwise evaluation has ~94–95% ties on GSM8K and ~78–81% on SVAMP; agreements with the exact verifier are largely driven by ties. Do not call this evidence of meaningful pairwise discrimination without qualifying it.
6. Near-zero or negative sampled-KL estimators are not exact nonnegative analytic KL. Follow the course's estimator and averaging convention.
7. Distinct saved adapters, nonzero finite LoRA tensors, and complete logs support completed training; without direct comparison against supplied midpoint weights, one cannot mathematically prove the exact magnitude of PPO/GRPO change from initialization.
8. The report has mandatory qualitative examples (reward-versus-quality, instruction compliance, GRPO bias, safety/judge disagreement, and Task 5 diagnostic failures); source JSONL exists, but examples have NOT been selected/verified for the paper. This is a remaining analysis task, not new training.

## Coverage and where to locate it

The `tables/` and `figures/` folders capture the saved quantitative comparisons from Tasks 1-5. Some required quantities (e.g., peak VRAM and wall-clock, exact task-specific qualitative examples, human judge validation) remain TODOs for the final main report; see the mandatory audit checklist. Additional visualization files are a curated catalog, and not every plot must be repeated in the first eight pages. The first eight pages must contain the quantities listed in the manual and can combine tables/figures. Numerical display alone cannot replace student-authored interpretation/qualitative examples.

The source file locations are named in `data/evidence_manifest.json`, `Report/README.md`, and the assignment-specific `results/task*/` JSON and JSONL outputs in the downloaded backups.

**Not covered:** Independent correctness proof for every objective formula, raw dataset provenance beyond the archived metadata, original midpoint-to-final tensor delta comparisons, and independent validation of the human labels. The present audit supports a no-retraining recommendation, not absolute proof of experiment correctness.

The main safety category figure gives full five-label distributions across all 18 categories and four policies (three nonzero labels and a legend noting the two zero labels). The manual audit is represented by aggregate counts only; raw labels remain in the local workbook.
