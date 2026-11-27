"""Profile prefix reuse, two attention calls, and fused MSA on Llama 3.1-8B."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def install_two_call_attention(backend=None):
    import torch
    if backend is None:
        import vllm.attention.backends.flash_attn as backend
    ordinary_attention = backend.flash_attn_varlen_func

    def two_calls(q, k, v, cu_seqlens_q, max_seqlen_q, max_seqlen_k,
                  seqused_k, seqlens_q_firstseg, seqlens_q_secondseg,
                  out=None, **kwargs):
        # The cache-rate profiler issues one request with one cached suffix.
        assert len(seqlens_q_firstseg) == 1
        assert int(seqlens_q_secondseg[0]) == 0
        first = int(seqlens_q_firstseg[0])
        output = out if out is not None else torch.empty_like(q)
        shared = {key: kwargs[key] for key in (
            "softmax_scale", "causal", "window_size", "alibi_slopes", "softcap", "block_table") if key in kwargs}
        for start, stop, key_length in ((0, first, int(seqused_k[0])),
                                        (first, q.shape[0], max_seqlen_k)):
            length = stop - start
            if not length:
                continue
            ordinary_attention(q=q[start:stop], k=k, v=v,
                cu_seqlens_q=torch.tensor([0, length], dtype=torch.int32, device=q.device),
                max_seqlen_q=length, max_seqlen_k=key_length,
                seqused_k=torch.tensor([key_length], dtype=torch.int32, device=q.device),
                out=output[start:stop], fa_version=2, **shared)
        return output

    backend.flash_attn_varlen_2seg_func = two_calls
    return two_calls


def profile(args):
    os.environ.update(VLLM_USE_V1="0", VLLM_USE_POD_ATTN="1" if args.mode == "msa" else "0",
                      VLLM_FLASH_ATTN_VERSION="2", VLLM_EVICTOR_POLICY="lru",
                      USE_MINEXP="0", USE_CONTINUUM="0", VLLM_SIMPLE_SHED_ALGO="0",
                      EVCITOR_BLOCK_PROTECT_FLAG="0")
    if args.mode == "two-kernel":
        install_two_call_attention()
    import numpy as np
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=1, enable_prefix_caching=True,
              collect_detailed_traces=["all"], max_model_len=20000, block_size=64,
              dtype="float16", gpu_memory_utilization=0.8)
    sampling = SamplingParams(temperature=0, max_tokens=1)
    total = 1024 if args.length == "1k" else 10240
    tail = args.tail_tokens if args.tail_tokens else (32 if args.length == "1k" else 128)
    repeats = args.repeats or (10 if args.length == "1k" else 6)
    base = [5] * (total - tail)
    manager = llm.llm_engine.scheduler[0].block_manager
    os.environ["EVCITOR_BLOCK_PROTECT_FLAG"] = "1"
    llm.generate(None, sampling_params=sampling, prompt_token_ids=base,
                 use_tqdm=False, slo_urgents=2.0)
    os.environ["EVCITOR_BLOCK_PROTECT_FLAG"] = "0"
    suffix = {"prefix": "", "two-kernel": "_2kernel", "msa": "_2seg"}[args.mode]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"profile_cache{suffix}_8B_{args.length}-tp1.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["cached_length", *[f"p{i + 1}" for i in range(repeats)], "AVG"])
        for point, cached in enumerate(range(48, total - tail + 1, args.step)):
            values = []
            for attempt in range(repeats):
                unique = 6 + point * repeats + attempt
                if args.mode == "prefix":
                    tokens = base[:cached] + [unique] * (total - cached)
                else:
                    manager.test_set_force_nocache(32, total - tail - cached + 32)
                    tokens = base + [unique] * tail
                result = llm.generate(None, sampling_params=sampling,
                                      prompt_token_ids=tokens, use_tqdm=False)
                values.append(result[0].metrics.model_forward_time)
            trimmed = sorted(values)[1:-1]
            writer.writerow([cached, *values, float(np.mean(trimmed))])
            stream.flush()
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "prefix", "two-kernel", "msa"], default="all")
    parser.add_argument("--length", choices=["all", "1k", "10k"], default="all")
    parser.add_argument("--model", default=os.environ.get("MODEL_8B_PATH", "meta-llama/Llama-3.1-8B-Instruct"))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("ARTIFACT_RESULTS", ROOT / "results")) / "raw/kernel")
    parser.add_argument("--step", type=int, default=16)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--tail-tokens", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.step <= 0 or (args.repeats is not None and args.repeats < 3):
        parser.error("Use a positive step and at least three repeats")
    if args.mode == "all" or args.length == "all" or args.dry_run:
        for mode in ("prefix", "two-kernel", "msa") if args.mode == "all" else [args.mode]:
            for length in ("1k", "10k") if args.length == "all" else [args.length]:
                command = [sys.executable, str(Path(__file__).resolve()), "--mode", mode,
                    "--length", length, "--model", args.model, "--output-dir", str(args.output_dir), "--step", str(args.step)]
                if args.repeats:
                    command += ["--repeats", str(args.repeats)]
                if args.tail_tokens:
                    command += ["--tail-tokens", str(args.tail_tokens)]
                print(subprocess.list2cmdline(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, check=True)
    else:
        profile(args)


if __name__ == "__main__":
    main()
