# ATML PA2 — NeurIPS 2026 report skeleton

**Status:** structure and numeric evidence only, **not a submitted report**. No AI-written abstract, analysis, discussion, or conclusion is included. Each `TODO` in `main.tex` must be written by the student, as the assignment explicitly prohibits GenAI report prose.

This project uses **the genuine `neurips_2026.sty`**, with `\usepackage[final]{neurips_2026}` to show the coursework author's name and avoid the unwanted "Preprint" footer. It is used as a formatting template, **not** a claim of NeurIPS publication or acceptance. The professor's **eight main-page** rule overrides the conference's own page count. References and optional appendix do not count, but no required evidence may be moved to the appendix.

## Build

From `Assignment-2/Report`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Or open the complete `Report` directory as an Overleaf project and compile with pdfLaTeX. `main.tex` is the root file. All included graphics are supplied in `figures/`; no GPU or Kaggle access is needed for report compilation.

Every figure and table uses `\begin{figure}[!htbp]` or `\begin{table}[!htbp]`. The `\ReportFigure` helper sets `[!htbp]` explicitly, and section barriers reduce cross-section movement. LaTeX still reserves the right to move a float when it cannot fit exactly; `!htbp` is a placement preference, not an absolute position command.

## Before submitting

- Write EVERY red `TODO` yourself, and check every citation.
- Keep the GitHub link at the end of the abstract.
- Inspect the final PDF visually and **count only main-body pages before references**. After adding prose, rebalance/reduce figures if main content exceeds 8 pages.
- Fill and independently validate Task 4 manual-audit result before citing its agreement; this skeleton intentionally contains no disputed manual labels.
- Inspect examples from original `results/task*_*/...examples.jsonl`, not made-up quotations.
- Review `EVIDENCE_AUDIT.md` and `data/evidence_manifest.json` before making claims.
- Do not mistake a signed sampled-response KL estimate near zero for proof that the policy is unchanged.

All attached numerical tables and figures are derived from the saved Task 1–5 backups, not generated from fake/demo data. `data/*.csv` contains small aggregate results for auditing, not raw course datasets. `data/evidence_manifest.json` stores source archive checksums.
