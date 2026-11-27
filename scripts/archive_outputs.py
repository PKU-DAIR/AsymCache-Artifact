"""Preserve replay files before running a launcher that appends output."""
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
raw_root = Path(os.environ.get("ARTIFACT_RESULTS", ROOT / "results")).resolve() / "raw"
folder = Path(sys.argv[1]).resolve()
if not folder.is_relative_to(raw_root):
    raise SystemExit(f"Expected an experiment directory under {raw_root}: {folder}")
for path in folder.rglob("*.jsonl"):
    destination = path.with_name(f"{path.name}.{time.time_ns()}.previous")
    path.rename(destination)
    print(f"Preserved {destination}")
