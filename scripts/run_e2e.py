"""Run the paper's LongBench/LooGLE matrix with the unified vLLM engine."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "vllm/examples/test/e2e-exp/reproduce"
POLICIES = {
    "LRU": ("lru", "0", "0"),
    "PENSIEVE": ("pensieve", "0", "0"),
    "BS": ("bs", "2", "0"),
    "HOTPREFIX": ("hotprefix", "0", "0"),
    "ASYMCACHE_NO_SCHEDULER": ("mwf", "2", "0"),
    "ASYMCACHE": ("mwf", "2", "1"),
}


def has_scheduler_comparison(config):
    return config["dataset"] == "longbench" and float(config["qps"]) == 0.04


def selected_baselines(config, selection):
    for baseline in POLICIES if selection == "ALL" else [selection]:
        if baseline == "ASYMCACHE_NO_SCHEDULER" and not has_scheduler_comparison(config):
            continue
        yield baseline


def configurations():
    for path in sorted((REPRO / "e2e/lru").glob("*/*_config*.txt")):
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            qps, p, grad, b, alpha, beta, dataset, model, frac = line.split()
            yield dict(qps=qps, p=p, grad=grad, b=b, alpha=alpha, beta=beta,
                       dataset=dataset, model=model, frac=frac,
                       interface=path.parent / "exp_interface1.py")


def wait_ready(server, port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError("vLLM exited; inspect server_output.log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError("vLLM startup timed out; inspect server_output.log")


def run(config, baseline, args):
    policy, minexp, scheduler = POLICIES[baseline]
    if baseline == "ASYMCACHE" and not has_scheduler_comparison(config):
        scheduler = "0"
    name = (f"300_{config['dataset']}_{baseline.lower()}_{config['model']}_"
            f"overall_qps{config['qps']}_frac{config['frac']}_CV0.25_gamma")
    results = Path(os.environ.get("ARTIFACT_RESULTS", ROOT / "results"))
    folder = results / "raw/vllm/e2e" / baseline.lower() / name
    model = os.environ.get(f"MODEL_{config['model']}_PATH",
                           f"meta-llama/Llama-3.1-{config['model']}-Instruct")
    port = str(args.port)
    output = folder / f"{name}.jsonl"
    env = os.environ.copy()
    env.update(VLLM_USE_V1="0", VLLM_USE_POD_ATTN="1", USE_CONTINUUM="0",
               VLLM_ALLOW_LONG_MAX_MODEL_LEN="1",
               DISABLE_MULTIPLIER="1", VLLM_EVICTOR_POLICY=policy,
               USE_MINEXP=minexp, VLLM_SIMPLE_SHED_ALGO=scheduler,
               VLLM_SIMPLE_SHED_ALGO_MAX_CHUNK_SIZE="2048", MINEXP_A="0",
               MINEXP_B=config["b"], MINEXP_ALPHA=config["alpha"],
               MINEXP_BETA=config["beta"], P=config["p"], GRAD=config["grad"],
               AGING_INTERVAL="2", CUDA_VISIBLE_DEVICES=args.gpu or (
                   "0" if config["model"] == "8B" else "0,1,2,3"),
               VLLM_PORT=str(args.port + 1000), VLLM_HOST_IP="127.0.0.1",
               CSET_DIR=str(ROOT / "vllm/csrc/c_set"),
               SCHEDULE_LOG_PATH=str(folder / "scheduler.log"),
               REPLAY_API_URL=f"http://127.0.0.1:{port}/v1/completions",
               REPLAY_MODEL=model,
               VLLM_EXPERIMENT_DATA_ROOT=os.environ.get(
                   "VLLM_EXPERIMENT_DATA_ROOT", str(ROOT / "data/workloads")))
    # Clear optional switches inherited from interactive benchmark sessions.
    env.pop("VLLM_ONLY_KEEP_SEG2", None)
    server_command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model, "-tp", "1" if config["model"] == "8B" else "4",
        "--port", port, "--block-size", "64", "--dtype", "float16",
        "--enable-prefix-caching", "--enable-chunked-prefill",
        "--max-model-len", "450000", "--max_num_batched_tokens", "8192",
        "--gpu-memory-utilization", "0.8"]
    replay_command = [sys.executable, str(config["interface"]), "--qps", config["qps"],
        "--frac", config["frac"], "--dataset", config["dataset"],
        "--model_size", config["model"], "--output_file", str(output)]
    metadata = {k: str(v) for k, v in config.items() if k != "interface"}
    metadata.update(baseline=baseline, server_command=server_command,
                    replay_command=replay_command, status="planned",
                    engine_settings={key: env[key] for key in (
                        "VLLM_EVICTOR_POLICY", "USE_MINEXP", "USE_CONTINUUM",
                        "VLLM_SIMPLE_SHED_ALGO", "VLLM_SIMPLE_SHED_ALGO_MAX_CHUNK_SIZE")})
    print(name, flush=True)
    if args.dry_run:
        return metadata
    schedule = Path(env["VLLM_EXPERIMENT_DATA_ROOT"]) / (
        f"{config['dataset']}_data/outputs_gamma_qps{config['frac']}_CV0.25_"
        f"{config['model']}_{config['qps']}_subset.csv")
    with schedule.open(encoding="utf-8-sig", newline="") as stream:
        expected = len(list(csv.DictReader(stream)))
    metadata["expected_requests"] = expected
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "run.json"
    if manifest.exists() and output.is_file() and json.loads(manifest.read_text())["status"] == "complete" and not args.rerun:
        print("  completed; skipping", flush=True)
        return metadata
    if output.exists():
        output.rename(folder / f"{output.name}.{time.time_ns()}.previous")
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    with (folder / "server_output.log").open("w") as log:
        server = subprocess.Popen(server_command, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_ready(server, port, args.startup_timeout)
            subprocess.run(replay_command, env=env, check=True)
            if not output.is_file() or not output.stat().st_size:
                raise RuntimeError(f"Replay produced no records: {output}")
            with output.open(encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            if len(records) != expected or any(not r.get("choices") or not r["choices"][0].get("metrics") for r in records):
                raise RuntimeError(f"Replay did not return {expected} complete metric records: {output}")
            metadata["status"] = "complete"
            manifest.write_text(json.dumps(metadata, indent=2) + "\n")
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=["ALL", *POLICIES], default="ALL")
    parser.add_argument("--workload", choices=["all", "longbench", "loogle"], default="all")
    parser.add_argument("--model-size", choices=["all", "8B", "70B"], default="all")
    parser.add_argument("--qps", type=float)
    parser.add_argument("--frac", choices=["5", "10"])
    parser.add_argument("--gpu", default=os.environ.get("GPU"))
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    count = 0
    for config in configurations():
        if args.workload not in ("all", config["dataset"]):
            continue
        if args.model_size not in ("all", config["model"]):
            continue
        if args.qps is not None and args.qps != float(config["qps"]):
            continue
        if args.frac is not None and args.frac != config["frac"]:
            continue
        for baseline in selected_baselines(config, args.baseline):
            run(config, baseline, args)
            count += 1
    if not count:
        parser.error("No experiment matched the selected configuration")
    print(f"Selected {count} experiments")


if __name__ == "__main__":
    main()
