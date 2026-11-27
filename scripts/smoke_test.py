"""Exercise the full replay-to-figure pipeline with synthetic timing records."""
import argparse
import csv
import contextlib
import importlib.util
import io
import json
import math
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plots"))
from collect_results import collect
from parse_data import load_data, load_data_tracea, load_data_bfcl, load_data_sense, validate_e2e


def check_e2e_launcher():
    from collections import Counter
    from run_e2e import configurations, run, selected_baselines

    args = argparse.Namespace(port=8050, gpu=None, dry_run=True)
    counts = Counter()
    with contextlib.redirect_stdout(io.StringIO()):
        for config in configurations():
            comparison = config["dataset"] == "longbench" and float(config["qps"]) == 0.04
            baselines = list(selected_baselines(config, "ALL"))
            assert "ASYMCACHE" in baselines
            assert ("ASYMCACHE_NO_SCHEDULER" in baselines) == comparison
            for baseline in baselines:
                metadata = run(config, baseline, args)
                expected = "1" if baseline == "ASYMCACHE" and comparison else "0"
                assert metadata["engine_settings"]["VLLM_SIMPLE_SHED_ALGO"] == expected
                counts[baseline] += 1
    assert sum(counts.values()) == 124
    assert counts["ASYMCACHE"] == 24 and counts["ASYMCACHE_NO_SCHEDULER"] == 4
    print("PASS: 124 launcher configurations and scheduler selections")


def record(index, sglang=False):
    arrival = index * 10.0
    result = {"sample_id": "same-session", "usage": {"prompt_tokens": 128, "completion_tokens": 4}}
    if sglang:
        result["meta_info"] = [dict(arrival_timestamp=arrival, first_token_timestamp=arrival + 2,
            last_token_timestamp=arrival + 4, scheduled_timestamp=arrival + 1, cached_tokens=64)]
    else:
        result["choices"] = [{"metrics": dict(arrival_time=arrival, first_token_time=arrival + 2,
            last_token_time=arrival + 4, first_scheduled_time=arrival + 1,
            suffix_computed_block_locs=[0, 0, 0, 0, 64])}]
    return result


def fixture(path, sglang=False):
    if os.name == "nt" and len(str(path.resolve())) > 240:
        short = hashlib.sha256(path.name.encode()).hexdigest()[:10]
        path = path.parent.parent / short / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record(i, sglang)) + "\n" for i in range(2)))


def generate(raw, kernel):
    for dataset in ("longbench", "loogle"):
        for model in ("8B", "70B"):
            for frac in (5, 10):
                for qps in (0.01, 0.02, 0.04):
                    for baseline in ("lru", "pensieve", "bs", "hotprefix", "asymcache_no_scheduler", "asymcache"):
                        if baseline == "asymcache_no_scheduler" and (dataset != "longbench" or qps != 0.04):
                            continue
                        name = f"300_{dataset}_{baseline}_{model}_overall_qps{qps}_frac{frac}_CV0.25_gamma"
                        fixture(raw / "vllm/e2e" / baseline / name / f"{name}.jsonl")
                    name = f"300_{dataset}_sglang_fcfs_{model}_overall_qps{qps}_frac{frac}_CV0.25_gamma"
                    fixture(raw / "sglang" / dataset / model / name / f"{name}.jsonl", True)
    for model in ("8B", "70B"):
        for qps in (0.2, 0.25, 0.3):
            for baseline in ("LRU", "PENSIEVE", "BS", "MWF", "HOTPREFIX", "sglang_fcfs"):
                name = f"traceA_2k_2100_{baseline}_{model}_qps{qps}_P75"
                backend = "sglang" if baseline == "sglang_fcfs" else "vllm"
                fixture(raw / backend / "tracea" / model / name / f"{name}.jsonl", backend == "sglang")
        for qps in (1.5, 1.75, 2.0):
            for baseline in ("LRU", "MWF", "CONTINUUM", "CONTINUUM_MWF"):
                name = f"2x_{baseline}_qps{qps}"
                fixture(raw / "vllm/agent" / model / name / f"2x_{baseline}_replay_output_qps{qps}.jsonl")
    config = ROOT / "vllm/examples/test/e2e-exp/hyper_parameter/ablation_config.txt"
    for line in config.read_text().splitlines():
        qps, point, grad, _, _, _, dataset, frac = line.split()
        name = f"LongBench_simulator_overall_qps{qps}_frac{frac}_set{point}_grad{grad}_gamma"
        fixture(raw / "vllm/sensitivity" / name / f"{name}.jsonl")
    kernel.mkdir(parents=True, exist_ok=True)
    for length in ("1k", "10k"):
        for mode in ("", "_2kernel", "_2seg"):
            with (kernel / f"profile_cache{mode}_8B_{length}-tp1.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["cached_length", "AVG"])
                writer.writerows([[64, 8.0], [256, 6.0], [512, 4.0]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("ARTIFACT_RESULTS", ROOT / "results")) / "smoke")
    args = parser.parse_args()
    check_e2e_launcher()
    run_dir = args.output_dir / uuid.uuid4().hex[:6]
    raw, summary = run_dir / "raw", run_dir / "summary"
    kernel = raw / "kernel"
    generate(raw, kernel)
    tables = collect(raw, summary)
    assert {k: len(v) for k, v in tables.items()} == {"e2e": 148, "tracea": 36, "agent": 24, "sensitivity": 20}
    for family, data in tables.items():
        for row in data:
            assert math.isclose(row["avg_ttft"], 2.0), (family, row)
            assert math.isclose(row["avg_tpot"], 1.0), (family, row)
            assert math.isclose(row["avg_hit_rate"], 0.5), (family, row)
            assert row["total_items"] == 2 and row["hit_items"] == 2
            if family == "agent":
                assert row["avg_job_latency"] == 14.0
                assert row["job_latency_p90"] == 14.0
    data = load_data(summary / "e2e.csv")
    for frac in ("5", "10"):
        validate_e2e(data, frac)
    load_data_tracea(summary / "tracea.csv")
    load_data_bfcl(summary / "agent.csv")
    panels = load_data_sense(summary / "sensitivity.csv", summary / "e2e.csv")
    assert [len(panels[i][0]) for i in range(1, 7)] == [5, 5, 4, 4, 4, 4]
    # Missing and duplicate baselines must not become zero-height plotted bars.
    duplicate = run_dir / "duplicate.csv"
    original = (summary / "e2e.csv").read_text()
    duplicate.write_text(original + original.splitlines()[1] + "\n")
    try:
        load_data(duplicate)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate result was accepted")
    del data["longbench"]["8B"]["5"]["lru"][0.01]
    try:
        validate_e2e(data, "5")
    except KeyError:
        pass
    else:
        raise AssertionError("Missing baseline was accepted")
    subprocess.run([sys.executable, str(ROOT / "plots/plot_all.py"), "--summary-dir", str(summary),
        "--kernel-dir", str(kernel), "--output-dir", str(run_dir / "figures")],
        env={**os.environ, "MPLBACKEND": "Agg"}, check=True)
    for name in ("figure_e2e_f5", "figure_e2e_f10", "figure_tracea", "sense", "figure_agent", "ablation-fused2"):
        for extension in ("pdf", "png"):
            path = run_dir / "figures" / f"{name}.{extension}"
            assert path.is_file() and path.stat().st_size > 1000, path
    (args.output_dir / "last_run.json").write_text(json.dumps({"directory": str(run_dir.resolve()), "status": "passed"}) + "\n")
    print("PASS: 228 synthetic experiments, metric checks, completeness checks and 12 figure files")


if __name__ == "__main__":
    main()
