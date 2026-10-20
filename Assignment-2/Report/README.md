# ATML PA2 — Complete experimental report

The NeurIPS 2026-styled `main.tex` includes a full abstract, introduction, method definitions, Tasks 1–6 evidence-based analysis, source-indexed response examples, conclusions, primary-paper citations, and an optional appendix.

The original uploaded assignment prohibited generative AI writing. The student subsequently reported a **direct professor exception** to that rule and to the eight-page restriction. Before submitting, verify the instructor's current permission and make sure it applies to this final report.

## Build

From `Assignment-2/Report/` run:

```bash
python scripts/verify_assets.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Alternatively, import the whole `Report/` folder into Overleaf and compile `main.tex`. The GitHub Actions workflow `.github/workflows/atml-pa2-report.yml` builds the PDF on each report push and uploads the `atml-pa2-neurips-final-report` PDF artifact.

**Figure/table placement:** all figure and table environments continue to use the requested `[!htbp]`. Float barriers follow each main report table and figure and keep evidence close to its source discussion. If an object cannot fit on a page, LaTeX may still place it on the next page; `[!htbp]` is a preference, not an exact fixed-placement command.

## Traceability and limitations

The numeric data and chart image files are unchanged from the archived Task 1–5 evidence. Source SHA-256 hashes live in `data/evidence_manifest.json`; the stored figure inventory is `FIGURE_INDEX.md`. The Task 4 full category classification heatmap and pooled five-label human/AI confusion matrix remain in the main paper; complementary classification maps and plots appear in `appendix.tex`.

Original qualitative examples from archived JSONL: DPO word-limit IDs `wl02` and `wl07`; PPO evaluation IDs `148` and `833`; GRPO IDs `708` and `288`; XSTest IDs `0` and `10`; math diagnostic problem `803`. Keep the report's caveats about PPO clipping, GRPO zero-token updates, safety-judge misclassification and Task 5 AI-judge ties.

The manual safety workbook has an archived checksum, but its human labels have not been independently re-adjudicated in this review. Independently verify those labels, the author details, the final PDF layout and the exception to the original assignment rules before LMS submission.
