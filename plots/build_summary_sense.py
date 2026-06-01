import re
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional


def parse_set_name(filename: str) -> Optional[str]:
    m = re.search(r"_set([^_]+)_grad", filename)
    return m.group(1) if m else None


def parse_grad(filename: str) -> Optional[str]:
    m = re.search(r"_grad([^_]+)_", filename)
    return m.group(1) if m else None


def parse_frac(filename: str) -> Optional[str]:
    m = re.search(r"_frac([^_]+)_", filename)
    return m.group(1) if m else None


def is_default_point(filename: str) -> bool:
    name = filename.lower()
    return ("cv0.25_p99" in name and "grad40" in name) or ("set1x1y_grad40" in name)


def sort_key(row: Dict[str, str], order: List[str]) -> int:
    filename = row.get("file", "")
    for i, token in enumerate(order):
        if token in filename:
            return i
    return len(order)


def load_rows(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return reader.fieldnames, rows


def build_sections(rows: List[Dict[str, str]], keep_lru: bool = True) -> Dict[str, List[Dict[str, str]]]:
    sections: Dict[str, List[Dict[str, str]]] = {
        "lru": [],
        "subplot1": [],
        "subplot2": [],
        "subplot3": [],
        "subplot4": [],
        "subplot5": [],
        "subplot6": [],
    }
    default_by_frac: Dict[str, Dict[str, str]] = {}
    for row in rows:
        filename = row.get("file", "")
        lower = filename.lower()
        frac = parse_frac(filename)
        set_name = parse_set_name(filename)
        grad = parse_grad(filename)
        if keep_lru and "lru" in lower:
            sections["lru"].append(row)
            continue
        if frac is None:
            continue
        if is_default_point(filename):
            default_by_frac[frac] = row
            continue
        if frac == "5":
            if set_name in {"0.6x", "0.8x", "1.2x", "1.4x"}:
                sections["subplot1"].append(row)
            elif set_name in {"0.80y", "0.90y", "0.95y"}:
                sections["subplot3"].append(row)
            elif set_name == "1x1y" and grad in {"10", "25", "50"}:
                sections["subplot5"].append(row)
        elif frac == "10":
            if set_name in {"0.6x", "0.8x", "1.2x", "1.4x"}:
                sections["subplot2"].append(row)
            elif set_name in {"0.80y", "0.90y", "0.95y"}:
                sections["subplot4"].append(row)
            elif set_name == "1x1y" and grad in {"10", "25", "50"}:
                sections["subplot6"].append(row)
    if "5" in default_by_frac:
        default = default_by_frac["5"]
        sections["subplot1"].append(default)
        sections["subplot3"].append(default)
        sections["subplot5"].append(default)
    if "10" in default_by_frac:
        default = default_by_frac["10"]
        sections["subplot2"].append(default)
        sections["subplot4"].append(default)
        sections["subplot6"].append(default)
    for key in ["subplot1", "subplot2"]:
        sections[key].sort(key=lambda r: sort_key(r, ["set0.6x", "set0.8x", "CV0.25_P99", "set1x1y_grad40", "set1.2x", "set1.4x"]))
    for key in ["subplot3", "subplot4"]:
        sections[key].sort(key=lambda r: sort_key(r, ["set0.80y", "set0.90y", "set0.95y", "CV0.25_P99", "set1x1y_grad40"]))
    for key in ["subplot5", "subplot6"]:
        sections[key].sort(key=lambda r: sort_key(r, ["set1x1y_grad10", "set1x1y_grad25", "CV0.25_P99", "set1x1y_grad40", "set1x1y_grad50"]))
    sections["lru"].sort(key=lambda r: sort_key(r, ["frac5", "frac10"]))
    return sections


def write_sectioned_csv(output: Path, headers: List[str], sections: Dict[str, List[Dict[str, str]]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ablation"])
        writer.writerow(headers)
        for row in sections["lru"]:
            writer.writerow([row.get(h, "") for h in headers])
        writer.writerow([])
        for name in ["subplot1", "subplot2", "subplot3", "subplot4", "subplot5", "subplot6"]:
            writer.writerow([name])
            for row in sections[name]:
                writer.writerow([row.get(h, "") for h in headers])
            writer.writerow([])


def print_summary(sections: Dict[str, List[Dict[str, str]]]) -> None:
    print("Output section sizes:")
    for name in ["lru", "subplot1", "subplot2", "subplot3", "subplot4", "subplot5", "subplot6"]:
        print(f"  {name}: {len(sections[name])} rows")
    expected = {"subplot1": 5, "subplot2": 5, "subplot3": 4, "subplot4": 4, "subplot5": 4, "subplot6": 4}
    for name, n_expected in expected.items():
        if len(sections[name]) != n_expected:
            print(
                f"[WARN] {name} has {len(sections[name])} rows, expected {n_expected} for the original 6-subplot figure. "
                "If the default/P99 row is missing, add the corresponding `CV0.25_P99_grad40` or `set1x1y_grad40` result row to the input CSV."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert flat ablation.csv into sectioned summary-sense.csv.")
    parser.add_argument("--input", "-i", required=True, help="Input flat ablation CSV.")
    parser.add_argument("--output", "-o", default="summary-sense.csv", help="Output sectioned CSV.")
    args = parser.parse_args()
    headers, rows = load_rows(Path(args.input))
    sections = build_sections(rows)
    write_sectioned_csv(Path(args.output), headers, sections)
    print_summary(sections)


if __name__ == "__main__":
    main()