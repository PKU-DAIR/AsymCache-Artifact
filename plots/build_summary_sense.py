"""Export the six sensitivity panels with their automatically selected references."""
import argparse
import csv
from pathlib import Path

from parse_data import load_data_sense
from plot_common import ROOT, RESULTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RESULTS / "summary/sensitivity.csv")
    parser.add_argument("--baselines", type=Path, default=RESULTS / "summary/e2e.csv")
    parser.add_argument("--output", type=Path, default=RESULTS / "summary/sensitivity-panels.csv")
    args = parser.parse_args()
    panels = load_data_sense(args.input, args.baselines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["panel", "point", "avg_ttft", "avg_tpot", "avg_hit_rate", "lru_ttft", "lru_tpot"])
        for panel in range(1, 7):
            reference = panels[0][(panel - 1) % 2]
            for point, values in enumerate(zip(*panels[panel]), 1):
                writer.writerow([panel, point, *values, reference["ttft"], reference["tpot"]])
    print(args.output)
