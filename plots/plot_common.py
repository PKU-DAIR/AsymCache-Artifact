"""Shared command-line paths and paper figure export settings."""
import argparse
from pathlib import Path

from collect_results import ROOT, RESULTS


def plot_args(table, baselines=False):
    parser = argparse.ArgumentParser(description=f"Plot the {table} paper results")
    parser.add_argument("--path", type=Path, default=RESULTS / "summary" / f"{table}.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    if baselines:
        parser.add_argument("--baselines", type=Path, default=RESULTS / "summary/e2e.csv")
    return parser.parse_args()


def save_figure(fig, directory, name, **kwargs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = directory / f"{name}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", **kwargs)
        print(path)
