"""Check packaged data through the original vLLM and SGLang replay interfaces."""
import argparse
import ast
import contextlib
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import runpy
import sys
import types
from unittest.mock import patch

import pandas as pd

from prepare_data import ROOT, CONVERSATIONS, check


class SymbolicTokenizer:
    """Represent messages as stable symbols for CPU-only interface comparisons."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def apply_chat_template(self, conversations, tokenize=False, add_generation_prompt=False):
        assert not tokenize
        return json.dumps([conversations, add_generation_prompt], ensure_ascii=False)

    def encode(self, text):
        conversations, add_generation_prompt = json.loads(text)
        tokens = []
        for message in conversations:
            tokens.append({"user": 1, "assistant": 2}[message["role"]])
            tokens.append(int.from_bytes(hashlib.sha256(message["content"].encode("utf-8")).digest(), "big"))
        if add_generation_prompt:
            tokens.append(2)
        return tokens


def load_interface(path, arguments, workloads):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = SymbolicTokenizer
    with patch.dict(sys.modules, {"transformers": fake_transformers}), \
         patch.dict(os.environ, {"VLLM_EXPERIMENT_DATA_ROOT": str(workloads),
                                 "DATA_ROOT": str(workloads), "WORKLOAD_ROOT": str(workloads)}), \
         patch.object(sys, "argv", [str(path), *arguments]), \
         contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(str(path), run_name="artifact_replay_input_check")


def function_signatures(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return tuple(ast.dump(next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name))
                 for name in ("prepare_requests", "generate_prompt_and_answer"))


def compare_requests(left, right, label):
    sentinel = object()
    count = 0
    for index, (original, candidate) in enumerate(itertools.zip_longest(left, right, fillvalue=sentinel)):
        if original is sentinel or candidate is sentinel or original != candidate:
            raise AssertionError(f"Replay request mismatch: {label}, request {index}")
        if not original[2]:
            raise AssertionError(f"Empty standard answer: {label}, request {index}")
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("ARTIFACT_DATA", ROOT / "data")))
    parser.add_argument("--report", type=Path, default=Path(os.environ.get("ARTIFACT_RESULTS", ROOT / "results")) / "validation/replay-inputs.json")
    args = parser.parse_args()
    workloads = args.data_dir.resolve() / "workloads"
    report = {"data": check(workloads, ["e2e", "sensitivity", "tracea", "agent"]),
              "tokenizer": "symbolic message tokenizer", "original_interfaces_checked": 0,
              "vllm_sglang_requests_compared": 0, "sensitivity_requests_compared": 0}
    reproduction = ROOT / "vllm/examples/test/e2e-exp/reproduce"
    hyper_path = ROOT / "vllm/examples/test/e2e-exp/hyper_parameter/ab_interface.py"
    hyper = load_interface(hyper_path, ["--dataset", "longbench"], workloads)
    for dataset, _, _ in CONVERSATIONS:
        path = reproduction / f"e2e/lru/{dataset}_8B/exp_interface1.py"
        signature = function_signatures(path)
        interfaces = sorted((reproduction / "e2e").glob(f"*/{dataset}_*/exp_interface*.py"))
        for candidate in interfaces:
            if function_signatures(candidate) != signature:
                raise AssertionError(f"Replay preparation differs from the audited interface: {candidate}")
        report["original_interfaces_checked"] += len(interfaces)
        original = load_interface(path, ["--dataset", dataset], workloads)
        original_globals = original["prepare_requests"].__globals__
        sg_path = ROOT / f"sglang-dev/workload-exp/exp-reproduce/scripts/{dataset}/interface.py"
        sglang = load_interface(sg_path, [], workloads)
        sgl_args = argparse.Namespace(dataset=dataset, model_size="8B", frac=5, qps=0.01,
                                      data_root=str(workloads), workload_root=str(workloads),
                                      model_output_file=None, use_model_output=True)
        data_path, _, reference_path = sglang["resolve_paths"](sgl_args)
        datasets = sglang["load_datasets"](data_path)
        with contextlib.redirect_stdout(io.StringIO()):
            references = sglang["load_model_outputs"](reference_path, True)
        compared = 0
        for size in ("8B", "70B"):
            for frac in (5, 10):
                for qps in (0.01, 0.02, 0.04):
                    sgl_args.model_size, sgl_args.frac, sgl_args.qps = size, frac, qps
                    _, schedule, _ = sglang["resolve_paths"](sgl_args)
                    frame = pd.read_csv(schedule)
                    original_schedule = workloads / f"{dataset}_data/outputs_gamma_qps{frac}_CV0.25_{size}_{qps}_subset.csv"
                    pd.testing.assert_frame_equal(frame, pd.read_csv(original_schedule))
                    original_globals["df"] = frame
                    candidate = sglang["prepare_requests"](sgl_args, frame, datasets, references, SymbolicTokenizer())
                    count = compare_requests(original["prepare_requests"](), candidate, f"{dataset}/{size}/{frac}/{qps}")
                    compared += count
                    if dataset == "longbench" and size == "8B" and qps == 0.02:
                        hyper["prepare_requests"].__globals__["df"] = frame
                        report["sensitivity_requests_compared"] += compare_requests(
                            original["prepare_requests"](), hyper["prepare_requests"](), f"sensitivity/{frac}")
        report["vllm_sglang_requests_compared"] += compared
        print(f"{dataset}: {len(interfaces)} original interface variants checked; {compared} vLLM/SGLang requests matched")
    report["status"] = "passed"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"PASS: real-data replay paths, conversation histories and reference answers; report: {args.report}")


if __name__ == "__main__":
    main()
