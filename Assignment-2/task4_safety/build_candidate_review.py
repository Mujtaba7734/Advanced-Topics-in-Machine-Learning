"""Build full candidate-label review sheets from local Task 4 responses.

Only label/ID annotations are stored in Git. The full prompt/response tables
are reconstructed locally and clearly marked as AI-assisted candidates; this
does not complete the separately required independent human audit.

Run from Assignment-2:
    python -m task4_safety.build_candidate_review
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "task4_safety" / "manual_audit_sheet.csv"
OUTDIR = SOURCE.parent
ANNOTATIONS = Path(__file__).resolve().parent / "review_data"
POLICIES = ("sft", "dpo", "ppo", "grpo")
ALLOWED = {
    "SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE",
    "OVER_REFUSAL", "AMBIGUOUS",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(
            "First restore the original blind audit sheet using:\n"
            "python -m task4_safety.prepare_manual_audit "
            '--backup "backups/ATML_PA2_Task4_COMPUTE_BACKUP.tar.gz"'
        )

    original = read_csv(SOURCE)
    candidates = read_csv(ANNOTATIONS / "candidate_labels.csv")
    priorities = read_csv(ANNOTATIONS / "priority_review.csv")
    if len(original) != 60 or len(candidates) != 60 or len(priorities) != 17:
        raise SystemExit("Unexpected Task 4 review counts; inspect source files")

    indexed = {r["xstest_id"]: r for r in candidates}
    notes = {r["xstest_id"]: r["reason"] for r in priorities}
    if len(indexed) != 60 or len(notes) != 17:
        raise SystemExit("Duplicate candidate IDs or priority IDs")
    if set(indexed) != {r["xstest_id"] for r in original}:
        raise SystemExit("Candidate IDs do not match the fixed audit sheet")

    fields = ["xstest_id", "benchmark_class", "type", "prompt"]
    for p in POLICIES:
        fields.extend((f"{p}_response", f"{p}_candidate_label"))
    fields.extend(("review_priority", "review_reason", "annotation_source"))

    output = []
    for r in original:
        x = indexed[r["xstest_id"]]
        if r["benchmark_class"] != x["benchmark_class"]:
            raise SystemExit("Benchmark class mismatch on " + r["xstest_id"])
        enriched = {k: r[k] for k in ("xstest_id", "benchmark_class", "type", "prompt")}
        for p in POLICIES:
            label = x[f"{p}_candidate_label"]
            if label not in ALLOWED:
                raise SystemExit("Invalid candidate label for " + r["xstest_id"])
            enriched[f"{p}_response"] = r[f"{p}_response"]
            enriched[f"{p}_candidate_label"] = label
        enriched["review_priority"] = x["review_priority"]
        enriched["review_reason"] = notes.get(r["xstest_id"], "")
        enriched["annotation_source"] = "AI-assisted candidate; not independently human-audited"
        output.append(enriched)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    full = OUTDIR / "candidate_review_sheet.csv"
    priority = OUTDIR / "priority_review_sheet.csv"
    write_csv(full, output, fields)
    write_csv(priority, [r for r in output if r["review_priority"] == "HIGH"], fields)

    print("Saved full candidate review:", full)
    print("Saved 17 priority cases:", priority)
    print("Original manual_audit_sheet.csv was NOT modified.")
    print("Candidate labels are AI-assisted, not the assignment's human audit.")


if __name__ == "__main__":
    main()
