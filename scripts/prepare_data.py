"""Import prepared conversations and check all workload inputs."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def check(workloads, groups):
    missing = []
    def require(path):
        if not path.is_file():
            missing.append(str(path))
            return False
        return True
    if "e2e" in groups or "sensitivity" in groups:
        for dataset, folder, context_key in [("longbench", "Longbench", "input"), ("loogle", "LooGLE_merged", "context")]:
            source_files = list((workloads / folder).glob("*.jsonl"))
            if not source_files:
                missing.append(str(workloads / folder / "*.jsonl"))
            conversations = {}
            for path in source_files:
                conversations[path.name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                for record in conversations[path.name]:
                    for key in (context_key, "instructions", "outputs"):
                        if key not in record:
                            raise ValueError(f"Missing {key}: {path}")
            reference = workloads / "data_preprocess" / f"{dataset}_deepseek_results.jsonl"
            outputs = set()
            if require(reference):
                for line in reference.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        record = json.loads(line)
                        outputs.add((record["file"], int(record["line"]), int(record["turn"])))
                        if "generate_text" not in record:
                            raise ValueError(f"Missing generate_text: {reference}")
            for size in ("8B", "70B"):
                for frac in (5, 10):
                    for qps in (0.01, 0.02, 0.04):
                        path = workloads / f"{dataset}_data/outputs_gamma_qps{frac}_CV0.25_{size}_{qps}_subset.csv"
                        if not require(path):
                            continue
                        with path.open(encoding="utf-8-sig", newline="") as stream:
                            reader = csv.DictReader(stream)
                            expected = {"Dataset Name", "Data Index", "Turn", "Timestamp"}
                            if not expected.issubset(reader.fieldnames or []):
                                raise ValueError(f"Invalid workload columns: {path}")
                            for row in reader:
                                name, index, turn = row["Dataset Name"], int(row["Data Index"]), int(row["Turn"])
                                if conversations and (name not in conversations or index >= len(conversations[name])):
                                    raise ValueError(f"Conversation index unavailable: {path}: {name}/{index}")
                                if outputs and any((name, index, previous) not in outputs for previous in range(turn + 1)):
                                    raise ValueError(f"Reference output unavailable: {path}: {name}/{index}/{turn}")
    if "tracea" in groups:
        require(workloads / "tracea/tracea_2k_cache_pressure_relaxed_2100.jsonl")
        require(workloads / "tracea/mapping_config.json")
    if "agent" in groups:
        for qps in (1.5, 1.75, 2.0):
            require(workloads / f"bfcl/2x_bfcl_web_search_prompts_qps{qps}.csv")
    if missing:
        raise FileNotFoundError("Import these workload files:\n" + "\n".join(missing))
    print("Workload input checks passed: " + ", ".join(groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Prepared dataset root containing Longbench/, LooGLE_merged/ and data_preprocess/")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("ARTIFACT_DATA", ROOT / "data")))
    parser.add_argument("--groups", nargs="+", choices=["e2e", "tracea", "agent", "sensitivity"], default=["e2e", "tracea", "agent", "sensitivity"])
    args = parser.parse_args()
    workloads = args.data_dir / "workloads"
    if args.source:
        for name in ("Longbench", "LooGLE_merged", "data_preprocess", "longbench_data", "loogle_data", "tracea", "bfcl"):
            source = args.source / name
            if source.is_dir() and source.resolve() != (workloads / name).resolve():
                shutil.copytree(source, workloads / name, dirs_exist_ok=True)
    check(workloads, args.groups)


if __name__ == "__main__":
    main()
