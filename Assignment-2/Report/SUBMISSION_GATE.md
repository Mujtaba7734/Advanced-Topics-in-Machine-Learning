# PA2 final verification gate (engineering checklist, not report prose)

**Status: not submission-ready.** The `Report/` folder is a structural LaTeX skeleton with evidence figures, numeric tables, and TODO prompts. It is not an eight-page finished paper. No AI-authored interpretation is provided.

## Confirmed from saved task backups
- Task 1: standard DPO one epoch, 94 steps; three beta short-run branches; length-controlled run, held-out strata and word-limit evidence.
- Task 2: standard PPO 20-update continuation; epsilon cached analysis and matched short forks; KL forks; training critic, reward, ratio and KL diagnostics.
- Task 3: standard GRPO 20 update slots, K-size probe at fixed generation budget, normalization comparison; four training steps had no usable tokens after truncated-output masking.
- Task 4: 450 common XSTest prompts each for SFT/DPO/PPO/GRPO; 1800 categorical AI-judge judgments. The completed 60-prompt workbook has 240 human labels and aggregate agreement/confusion counts in `data/15_manual_audit.json`. Review the labels independently before interpreting them. Automated judge's zero OVER_REFUSAL rate is not trustworthy without independent checking.
- Task 5: 300 GSM8K and 100 SVAMP fixed examples per policy; 20 problems x five controlled answers; 80 preference comparisons with AI and outcome-based scoring.
- Task 6: interpretive synthesis only; no new training specified.

## Known analysis cautions
- PPO clipping fractions are zero in the recorded standard training log. Do not claim large nonzero on-policy clipping based on the separate cached-token probe.
- PPO epsilon short forks show identical held-out metrics; do not invent an epsilon effect.
- Four GRPO updates have zero optimizable tokens; do not call all 20 effective gradient updates.
- Large AI-judge tie rates in Task 5 make its discriminative power limited.
- Standard one-epoch DPO is NOT compute-matched to the beta short forks.
- The original starting PPO/GRPO midpoint adapters have not been directly compared tensor-by-tensor to the saved final adapters. Adapter presence, finite/nonzero values and training logs support completion but cannot prove the exact update delta.
- Task-specific qualitative failures and favorable counterexamples still need manual selection from saved records; do not fabricate examples.

## Mandatory before LMS submission
1. Write the abstract, figure captions, interpretation, synthesis and conclusions entirely yourself, following PA2 AI restrictions.
2. Replace every TODO, insert grounded qualitative examples and actual references.
3. Count **main-text** pages after writing. The initial placeholder preview is 11 pages including supplements, not a certified eight-main-page submission. Required Evidence must be within the eight main pages.
4. Verify `neurips_2026.sty` against the professor's course-supplied style if one was provided. Do not substitute a temporary local geometry harness.
5. Run `python scripts/verify_assets.py` and `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` inside `Assignment-2/Report`. Inspect the PDF for truncated plots and floating placement. `[!htbp]` is a LaTeX preference, not a strict guarantee.
6. Validate that every reported metric has a source in `data/evidence_manifest.json` and archived JSON/JSONL results. Keep raw XSTest prompts and large model weights out of public Git.
7. Freeze Git commit SHA and submit the final PDF and repository URL to LMS.

**Retraining decision:** No currently established flaw warrants automatic retraining, but a complete proof of algorithmic and data correctness has not been performed. Honest limitation reporting is required.
