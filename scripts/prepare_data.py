"""Validate packaged workloads and import prepared replay inputs."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
CONVERSATIONS = (("longbench", "Longbench", "input"), ("loogle", "LooGLE_merged", "context"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record: {path}:{index + 1}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object: {path}:{index + 1}")
            yield record


def load_conversations(workloads, dataset, folder, context_key):
    datasets = {}
    for path in sorted((workloads / folder).glob("*.jsonl")):
        records = list(read_jsonl(path))
        if not records:
            raise ValueError(f"Empty conversation file: {path}")
        for index, record in enumerate(records):
            instructions, outputs = record.get("instructions"), record.get("outputs")
            valid = (isinstance(record.get(context_key), str)
                     and isinstance(instructions, list) and isinstance(outputs, list)
                     and len(instructions) > 0 and len(instructions) == len(outputs)
                     and all(isinstance(value, str) for value in instructions + outputs))
            if not valid:
                raise ValueError(f"Invalid {dataset} conversation schema: {path}:{index + 1}")
        datasets[path.name] = records
    if not datasets:
        raise FileNotFoundError(f"No conversation JSONL files: {workloads / folder}")
    return datasets


def load_references(workloads, dataset, datasets):
    path = workloads / "data_preprocess" / f"{dataset}_deepseek_results.jsonl"
    references = {}
    for index, record in enumerate(read_jsonl(path)):
        key = (record.get("file"), record.get("line"), record.get("turn"))
        name, line, turn = key
        if (not isinstance(name, str) or type(line) is not int or type(turn) is not int
                or name not in datasets or not 0 <= line < len(datasets[name])
                or not 0 <= turn < len(datasets[name][line]["instructions"])
                or not isinstance(record.get("generate_text"), str)):
            raise ValueError(f"Invalid reference output: {path}:{index + 1}")
        if key in references:
            raise ValueError(f"Duplicate reference output: {path}: {key}")
        references[key] = record["generate_text"]
    if not references:
        raise ValueError(f"Empty reference outputs: {path}")
    return references


def check_conversations(workloads, dataset, folder, context_key):
    datasets = load_conversations(workloads, dataset, folder, context_key)
    references = load_references(workloads, dataset, datasets)
    schedules = requests = 0
    required = set()
    for size in ("8B", "70B"):
        for frac in (5, 10):
            for qps in (0.01, 0.02, 0.04):
                path = workloads / f"{dataset}_data/outputs_gamma_qps{frac}_CV0.25_{size}_{qps}_subset.csv"
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream)
                    expected = {"Dataset Name", "Data Index", "Turn", "Timestamp"}
                    if not expected.issubset(reader.fieldnames or []):
                        raise ValueError(f"Invalid workload columns: {path}")
                    count, last_timestamp = 0, -1.0
                    for row in reader:
                        name, index, turn = row["Dataset Name"], int(row["Data Index"]), int(row["Turn"])
                        timestamp = float(row["Timestamp"])
                        if not math.isfinite(timestamp) or timestamp < 0 or timestamp < last_timestamp:
                            raise ValueError(f"Invalid replay timestamp: {path}: row {count + 2}")
                        if name not in datasets or not 0 <= index < len(datasets[name]):
                            raise ValueError(f"Conversation index unavailable: {path}: {name}/{index}")
                        if not 0 <= turn < len(datasets[name][index]["instructions"]):
                            raise ValueError(f"Turn index unavailable: {path}: {name}/{index}/{turn}")
                        for previous in range(turn + 1):
                            key = (name, index, previous)
                            if key not in references:
                                raise ValueError(f"Reference output unavailable: {path}: {key}")
                            required.add(key)
                        count += 1
                        last_timestamp = timestamp
                if not count:
                    raise ValueError(f"Empty replay schedule: {path}")
                schedules += 1
                requests += count
    report = dict(conversation_files=len(datasets),
                  conversations=sum(map(len, datasets.values())),
                  conversation_turns=sum(len(record["instructions"]) for records in datasets.values() for record in records),
                  reference_outputs=len(references), schedules=schedules, scheduled_requests=requests,
                  required_reference_outputs=len(required))
    print(f"{dataset}: {report['conversations']} conversations, {len(references)} reference outputs, "
          f"{schedules} schedules, {requests} scheduled requests; all replayed turns covered")
    return report


def check(workloads, groups):
    report = {}
    if "e2e" in groups or "sensitivity" in groups:
        for dataset, folder, context_key in CONVERSATIONS:
            report[dataset] = check_conversations(workloads, dataset, folder, context_key)
    required = []
    if "tracea" in groups:
        required += [workloads / "tracea/tracea_2k_cache_pressure_relaxed_2100.jsonl",
                     workloads / "tracea/mapping_config.json"]
    if "agent" in groups:
        required += [workloads / f"bfcl/2x_bfcl_web_search_prompts_qps{qps}.csv" for qps in (1.5, 1.75, 2.0)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing workload files:\n" + "\n".join(missing))
    print("Workload input checks passed: " + ", ".join(groups))
    return report


def import_workloads(source, workloads):
    files = []
    for dataset, folder, _ in CONVERSATIONS:
        files.extend((source / folder).glob("*.jsonl"))
        reference = source / "data_preprocess" / f"{dataset}_deepseek_results.jsonl"
        if reference.is_file():
            files.append(reference)
        files.extend((source / f"{dataset}_data").glob("outputs_gamma_qps*_subset.csv"))
    for relative in ["tracea/tracea_2k_cache_pressure_relaxed_2100.jsonl", "tracea/mapping_config.json",
                     *[f"bfcl/2x_bfcl_web_search_prompts_qps{qps}.csv" for qps in (1.5, 1.75, 2.0)]]:
        if (source / relative).is_file():
            files.append(source / relative)
    for path in sorted(files):
        destination = workloads / path.relative_to(source)
        if path.resolve() == destination.resolve():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        print(f"Imported {path.relative_to(source)}")


def write_manifest(data_dir):
    manifest = {}
    for path in sorted((data_dir / "workloads").rglob("*")):
        if path.is_file():
            manifest[path.relative_to(data_dir).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = data_dir / "workloads.sha256.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Recorded SHA-256 hashes for {len(manifest)} workload files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Import prepared conversations, reference outputs and replay inputs")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("ARTIFACT_DATA", ROOT / "data")))
    parser.add_argument("--groups", nargs="+", choices=["e2e", "tracea", "agent", "sensitivity"],
                        default=["e2e", "tracea", "agent", "sensitivity"])
    parser.add_argument("--write-manifest", action="store_true", help="Refresh workload SHA-256 hashes after validation")
    args = parser.parse_args()
    workloads = args.data_dir / "workloads"
    if args.source:
        import_workloads(args.source, workloads)
    check(workloads, args.groups)
    if args.write_manifest:
        write_manifest(args.data_dir)


if __name__ == "__main__":
    main()
