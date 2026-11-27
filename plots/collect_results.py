"""Summarize replay JSONL files into named CSV tables for all paper plots."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("ARTIFACT_RESULTS", ROOT / "results"))
FIELDS = ["dataset", "model", "frac", "approach", "qps", "set", "grad", "file",
          "relative_path", "total_items", "avg_ttft", "avg_tpot", "hit_items",
          "avg_hit_rate", "total_time", "throughput", "avg_job_latency", "job_latency_p90"]
ALIASES = {"pensieve": "pensi", "pensi": "pensi", "mwf": "minexp", "minexp": "minexp",
           "asymcache_no_scheduler": "minexp", "asymcache": "shed", "max_score": "bs",
           "sglang_fcfs": "sglang"}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(text, family=None):
    value = text.lower().replace("\\", "/")
    model = re.search(r"(?:^|[/_])(70b|8b)(?:[/_.]|$)", value)
    qps = re.search(r"qps([0-9]+(?:\.[0-9]+)?)", value)
    frac = re.search(r"frac(\d+)", value)
    dataset_match = re.search(r"(?:^|[/_])(longbench|loogle|tracea)(?:[/_]|$)", value)
    dataset = dataset_match.group(1) if dataset_match else ""
    if family == "agent":
        dataset = "agent"
    approach = ""
    for token in ["continuum_mwf", "continuum", "asymcache_no_scheduler", "asymcache",
                  "sglang_fcfs", "sglang", "hotprefix", "max_score", "pensieve", "pensi",
                  "minexp", "mwf", "shed", "bs", "lru"]:
        if re.search(r"(?:^|[/_])" + token + r"(?:[/_ .]|$)", value):
            approach = ALIASES.get(token, token)
            break
    if family == "agent" and approach == "minexp":
        approach = "mwf"
    set_match = re.search(r"_set([^_]+)_grad", value)
    grad = re.search(r"_grad(\d+)", value)
    return dict(dataset=dataset, model=model.group(1).upper() if model else (
        "8B" if family == "sensitivity" else ""), qps=float(qps.group(1)) if qps else "",
        frac=frac.group(1) if frac else "", approach=approach,
        set=set_match.group(1) if set_match else "", grad=grad.group(1) if grad else "")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def collect(raw, output):
    e2e = load_module("artifact_e2e_stats", ROOT / "vllm/examples/test/e2e-exp/reproduce/result_analyse.py")
    tracea = load_module("artifact_tracea_stats", ROOT / "vllm/examples/test/e2e-exp/Qwen_trace/auto_script/result_analyse.py")
    agent = load_module("artifact_agent_stats", ROOT / "vllm/examples/test/e2e-exp/BFCL_workload_reproduce/full_result.py")
    sglang = load_module("artifact_sglang_stats", ROOT / "sglang-dev/workload-exp/exp-reproduce/traceA_analyse.py")
    tables = {key: [] for key in ("e2e", "tracea", "agent", "sensitivity")}
    for path in sorted(raw.rglob("*.jsonl")):
        relative = path.relative_to(raw)
        parts = [p.lower() for p in relative.parts]
        if path.name in {"engine_request_outputs.jsonl", "replay_summary.jsonl"}:
            continue
        if "sensitivity" in parts:
            family, analyzer = "sensitivity", e2e
        elif "agent" in parts and "replay_output" in path.name:
            family, analyzer = "agent", agent
        elif "tracea" in parts:
            family, analyzer = "tracea", sglang if "sglang" in parts else tracea
        elif "e2e" in parts or ("sglang" in parts and any(d in parts for d in ("longbench", "loogle"))):
            family, analyzer = "e2e", sglang if "sglang" in parts else e2e
        else:
            continue
        meta = identity(str(relative), family)
        manifest = path.parent / "run.json"
        if manifest.exists():
            run = json.loads(manifest.read_text())
            if run.get("status") != "complete":
                raise ValueError(f"Incomplete experiment: {manifest}")
            meta.update(dataset=run["dataset"], model=run["model"], qps=float(run["qps"]),
                        frac=run["frac"], approach=ALIASES.get(run["baseline"].lower(), run["baseline"].lower()))
        if family == "sensitivity":
            meta.update(dataset="longbench", approach="minexp")
        if not meta["dataset"] or not meta["model"] or meta["qps"] == "" or not meta["approach"]:
            raise ValueError(f"Cannot identify experiment: {relative}")
        stats = analyzer.analyze_jsonl(path)
        if not stats["total"]:
            raise ValueError(f"No valid replay metrics: {path}")
        with path.open(encoding="utf-8") as stream:
            record_count = sum(bool(line.strip()) for line in stream)
        if stats["total"] != record_count:
            raise ValueError(f"Replay contains records without complete metrics: {path}")
        if manifest.exists() and stats["total"] != run.get("expected_requests", stats["total"]):
            raise ValueError(f"Replay request count differs from its manifest: {path}")
        row = {key: stats.get(key, "") for key in FIELDS}
        row.update(meta, file=path.name, relative_path=relative.as_posix(),
                   total_items=stats["total"], hit_items=stats["hit_total"])
        tables[family].append(row)
    for family, rows in tables.items():
        write_csv(output / f"{family}.csv", rows)
        print(f"{family}: {len(rows)} experiments -> {output / (family + '.csv')}")
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RESULTS / "raw")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "summary")
    args = parser.parse_args()
    if not args.raw_root.is_dir():
        parser.error(f"Result directory does not exist: {args.raw_root}")
    collect(args.raw_root, args.output_dir)
