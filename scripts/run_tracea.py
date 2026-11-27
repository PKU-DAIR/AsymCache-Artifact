"""Calculate each QPS's MWF parameters and launch the TraceA matrix."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=["all", "8B", "70B"], default="all")
    parser.add_argument("--qps", type=float, choices=[0.2, 0.25, 0.3])
    parser.add_argument("--baseline", choices=["ALL", "LRU", "PENSIEVE", "BS", "MWF", "HOTPREFIX"], default="ALL")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    trace_dir = ROOT / "vllm/examples/test/e2e-exp/Qwen_trace"
    data = Path(os.environ.get("ARTIFACT_DATA", ROOT / "data"))
    results = Path(os.environ.get("ARTIFACT_RESULTS", ROOT / "results"))
    input_file = data / "workloads/tracea/tracea_2k_cache_pressure_relaxed_2100.jsonl"
    for qps in [args.qps] if args.qps else [0.2, 0.25, 0.3]:
        prefix = results / "parameters" / f"tracea_qps{qps}"
        subprocess.run([sys.executable, str(trace_dir / "CDF.py"), str(input_file),
                        "--target-qps", str(qps), "--percentile", "75",
                        "--output-prefix", str(prefix), "--no-plot"], check=True)
        params = json.loads(Path(f"{prefix}.json").read_text())["mwf_parameters"]
        for size in ["8B", "70B"] if args.model_size == "all" else [args.model_size]:
            command = ["bash", str(trace_dir / "auto_script/run_traceA_2100.sh"),
                "--model-size", size, "--qps", str(qps), "--input-file", str(input_file),
                "--tracea-dir", str(trace_dir / "auto_script"),
                "--agent-script", str(trace_dir / "auto_script/replay_traceA.py"),
                "--result-root", str(results / "raw/vllm/tracea" / size),
                "--minexp-b", str(params["MINEXP_B"]),
                "--minexp-alpha", str(params["MINEXP_ALPHA"]),
                "--minexp-beta", str(params["MINEXP_BETA"])]
            command += ["--all-policies"] if args.baseline == "ALL" else ["--baseline", args.baseline]
            if args.dry_run:
                command.append("--dry-run")
            env = os.environ.copy()
            env.setdefault("VLLM_HOST_IP", "127.0.0.1")
            env["CUDA_VISIBLE_DEVICES"] = os.environ.get(f"GPU_{size}", "0" if size == "8B" else "0,1,2,3")
            env["TP"] = "1" if size == "8B" else "4"
            subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
