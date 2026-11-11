"""Disk retention helpers; no changes to the experiment or recovery format."""
import os
from pathlib import Path


def prune_older_recoveries(current, latest):
    current, latest = Path(current), Path(latest)
    # Never remove fallback checkpoints unless latest references the committed file.
    if not current.is_file() or not latest.is_file() or not os.path.samefile(current, latest):
        return []
    removed = []
    for old in sorted(current.parent.glob("recovery_*.pt")):
        if old.name < current.name:
            old.unlink()
            removed.append(old.name)
    return removed
