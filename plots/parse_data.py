"""Read named result columns and adapt them to the original paper layouts."""
import csv
import math
from pathlib import Path

from collect_results import ROOT, RESULTS, identity


def number(value):
    text = str(value).strip()
    return float(text[:-1]) / 100 if text.endswith("%") else float(text)


def rows(path):
    header = None
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for values in csv.reader(stream):
            if "avg_ttft" in values:
                header = [v.strip() for v in values]
                continue
            if not header or not values:
                continue
            row = dict(zip(header, values))
            if not row.get("avg_ttft", "").strip():
                continue
            try:
                number(row["avg_ttft"])
            except ValueError:
                continue
            meta = identity(row.get("relative_path", "") + "/" + row.get("file", ""))
            for key in meta:
                if row.get(key, "") != "":
                    meta[key] = row[key]
            yield {**row, **meta}


def metrics(row):
    result = dict(ttft=number(row["avg_ttft"]), tpot=number(row["avg_tpot"]),
                hit_items=int(row["hit_items"]), hit_rate=number(row["avg_hit_rate"]),
                total_items=int(row.get("total_items", 300)),
                total_time=number(row.get("total_time", 0)),
                tpt=number(row.get("throughput", 0)))
    if not all(math.isfinite(result[key]) and result[key] >= 0 for key in ("ttft", "tpot", "hit_rate")):
        raise ValueError(f"Invalid metrics: {row['file']}")
    if result["hit_rate"] > 1 or result["hit_items"] > result["total_items"]:
        raise ValueError(f"Invalid hit rate or request count: {row['file']}")
    return result


def insert(target, key, value, description):
    if key in target:
        raise ValueError(f"Duplicate result: {description}")
    target[key] = value


def e2e_approach(row):
    approach = row["approach"]
    if approach == "shed" and (row["dataset"] != "longbench" or number(row["qps"]) != 0.04):
        return "minexp"
    return approach


def load_data(path=RESULTS / "summary/e2e.csv", v2=False):
    dataset = {d: {m: {} for m in ("8B", "70B")} for d in ("longbench", "loogle")}
    for row in rows(path):
        d, model, frac = [row[k] for k in ("dataset", "model", "frac")]
        approach = e2e_approach(row)
        if d not in dataset:
            continue
        qps = number(row["qps"])
        target = dataset[d][model].setdefault(str(frac), {}).setdefault(approach, {})
        insert(target, qps, metrics(row), row["file"])
    return dataset


def validate_e2e(dataset, frac):
    for d in ("longbench", "loogle"):
        for model in ("8B", "70B"):
            for approach in ("lru", "sglang", "bs", "pensi", "hotprefix", "minexp"):
                for qps in (0.01, 0.02, 0.04):
                    dataset[d][model][frac][approach][qps]
            if d == "longbench":
                dataset[d][model][frac]["shed"][0.04]


def load_data_tracea(path=RESULTS / "summary/tracea.csv"):
    dataset = {"tracea": {m: {} for m in ("8B", "70B")}}
    for row in rows(path):
        target = dataset["tracea"][row["model"]].setdefault(row["approach"], {})
        insert(target, number(row["qps"]), metrics(row), row["file"])
    for model in ("8B", "70B"):
        for approach in ("lru", "sglang", "bs", "pensi", "hotprefix", "minexp"):
            for qps in (0.2, 0.25, 0.3):
                dataset["tracea"][model][approach][qps]
    return dataset


def load_data_bfcl(path=RESULTS / "summary/agent.csv"):
    dataset = {m: {} for m in ("8b", "70b")}
    for row in rows(path):
        qps = str(float(row["qps"]))
        approach = row["approach"]
        if approach == "minexp":
            approach = "mwf"
        target = dataset[row["model"].lower()].setdefault(qps, {})
        value = metrics(row)
        value.update(job_avg=number(row["avg_job_latency"]),
                     job_p90=number(row["job_latency_p90"]), hit_rate=value["hit_rate"] * 100)
        insert(target, approach, value, row["file"])
    for model in dataset:
        for qps in ("1.5", "1.75", "2.0"):
            for approach in ("lru", "mwf", "continuum", "continuum_mwf"):
                dataset[model][qps][approach]
    return dataset


def load_data_sense(path=RESULTS / "summary/sensitivity.csv", baseline_path=RESULTS / "summary/e2e.csv"):
    points = {}
    for row in rows(path):
        key = (str(row["frac"]), row.get("set", ""), str(row.get("grad", "")))
        insert(points, key, metrics(row), row["file"])
    defaults, references = {}, {}
    for row in rows(baseline_path):
        if row["dataset"] == "longbench" and row["model"] == "8B" and number(row["qps"]) == 0.02:
            if row["approach"] == "lru":
                references[str(row["frac"])] = metrics(row)
            if e2e_approach(row) == "minexp":
                defaults[str(row["frac"])] = metrics(row)
    for frac in ("5", "10"):
        if (frac, "1x1y", "40") in points:
            defaults[frac] = points[(frac, "1x1y", "40")]
    result = {0: [references[f] for f in ("5", "10")]}
    for index in range(1, 7):
        frac = "5" if index % 2 else "10"
        if index <= 2:
            selected = [points[(frac, s, "40")] if s else defaults[frac]
                        for s in ("0.6x", "0.8x", "", "1.2x", "1.4x")]
        elif index <= 4:
            selected = [points[(frac, s, "40")] for s in ("0.80y", "0.90y", "0.95y")] + [defaults[frac]]
        else:
            selected = [defaults[frac] if grad == "40" else points[(frac, "1x1y", grad)]
                        for grad in ("10", "25", "40", "50")]
        result[index] = [[p[key] for p in selected] for key in ("ttft", "tpot", "hit_rate")]
    return result
