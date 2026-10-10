"""Restore the fixed blind Task 4 worksheet from the verified Kaggle backup.

The full worksheet contains fixed course prompts and generated responses; it is
created locally from the research backup rather than committed as raw data.
This script uses Python's standard library only (no GPU or ML dependencies).
"""
from __future__ import annotations

import argparse
import csv
import io
import tarfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BACKUP_NAME = "ATML_PA2_Task4_COMPUTE_BACKUP.tar.gz"
MEMBER = "results/task4_safety/manual_audit_sheet.csv"
OUTPUT = REPO / MEMBER
POLICIES = ("sft", "dpo", "ppo", "grpo")
LABEL_FIELDS = tuple(f"{policy}_manual_label" for policy in POLICIES)


def find_backup() -> Path | None:
    home = Path.home()
    candidates = (
        Path.cwd(),
        REPO,
        REPO.parent,
        home / "Downloads",
        home / "OneDrive" / "Downloads",
        home / "Desktop",
    )
    for folder in candidates:
        candidate = folder / BACKUP_NAME
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--backup", type=Path, default=None,
        help="Path to the Task 4 compute backup .tar.gz (normally in Downloads)",
    )
    args = ap.parse_args()

    if OUTPUT.is_file():
        print("Existing manual audit CSV preserved; no labels were overwritten:")
        print(OUTPUT)
        print("Open this file in VS Code to continue reviewing.")
        return

    backup = args.backup or find_backup()
    if backup is None:
        print(f"Couldn't find {BACKUP_NAME} in Downloads or the repository.")
        entered = input("Paste the full path to the downloaded Task 4 backup: ").strip()
        backup = Path(entered.strip('"').strip("'"))
    backup = backup.expanduser()
    if not backup.is_file():
        raise SystemExit(f"Task 4 backup not found: {backup}")

    # Read exactly one approved file. Do not extract arbitrary tar paths.
    with tarfile.open(backup, "r:gz") as archive:
        try:
            member = archive.getmember(MEMBER)
        except KeyError as exc:
            raise SystemExit(f"Backup missing {MEMBER}") from exc
        source = archive.extractfile(member)
        if source is None:
            raise SystemExit(f"Cannot read backup member {MEMBER}")
        raw = source.read()

    text = raw.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    if len(rows) != 60:
        raise SystemExit(f"Expected 60 blind audit prompts; found {len(rows)}")

    counts = Counter(r.get("benchmark_class", "").upper() for r in rows)
    if counts != Counter({"SAFE": 30, "UNSAFE": 30}):
        raise SystemExit(f"Expected 30 SAFE and 30 UNSAFE prompts; got {counts}")

    if any(field not in rows[0] for field in LABEL_FIELDS):
        raise SystemExit("Missing one or more required manual label columns")
    if any(r.get(field, "").strip() for r in rows for field in LABEL_FIELDS):
        raise SystemExit("Backup contains nonblank manual labels; inspect before proceeding")

    source_ids = [r["xstest_id"] for r in rows]
    if len(set(source_ids)) != 60:
        raise SystemExit("Duplicate audit prompt IDs in backup")
    if any("ai_label" in key.lower() for key in rows[0]):
        raise SystemExit("Audit sheet is not blind: unexpected AI-label column")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(raw)
    print("Restored blind Task 4 manual audit worksheet:")
    print(OUTPUT)
    print("60 fixed prompts | 30 SAFE + 30 UNSAFE | 240 empty manual labels")
    print("Open the CSV in VS Code, fill the four *_manual_label columns, and save.")
    print("Keep course judge and AI candidate-label files closed while judging.")


if __name__ == "__main__":
    main()
