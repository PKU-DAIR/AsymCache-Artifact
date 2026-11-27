"""Render the paper figures from experiment summaries or paper reference data."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("ARTIFACT_RESULTS", ROOT / "results"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, default=RESULTS / "summary")
    parser.add_argument("--kernel-dir", type=Path, default=RESULTS / "raw/kernel")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--reference", action="store_true", help="Use data/reference/paper")
    parser.add_argument("--figures", nargs="+", choices=["e2e", "tracea", "sensitivity", "agent", "kernel"],
                        default=["e2e", "tracea", "sensitivity", "agent", "kernel"])
    args = parser.parse_args()
    if args.reference:
        args.summary_dir = ROOT / "data/reference/paper"
        args.kernel_dir = ROOT / "data/reference/paper/kernel"
    env = {**os.environ, "MPLBACKEND": "Agg"}
    def plot(script, *options):
        subprocess.run([sys.executable, str(ROOT / "plots" / script),
                        "--output-dir", str(args.output_dir), *map(str, options)], env=env, check=True)
    for figure in args.figures:
        if figure == "e2e":
            for dispersion in ("low", "high"):
                plot("plot-e2e.py", "--path", args.summary_dir / "e2e.csv", "--div", dispersion)
        elif figure == "kernel":
            plot("plot-kernel.py", "--data-dir", args.kernel_dir)
        elif figure == "sensitivity":
            plot("plot_sense.py", "--path", args.summary_dir / "sensitivity.csv",
                 "--baselines", args.summary_dir / "e2e.csv")
        else:
            plot(f"plot-{'agent' if figure == 'agent' else 'tracea'}.py",
                 "--path", args.summary_dir / f"{figure}.csv")


if __name__ == "__main__":
    main()
