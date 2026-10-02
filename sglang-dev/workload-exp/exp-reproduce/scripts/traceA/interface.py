import os
import time
import tqdm
import glob
import json
import asyncio
import aiohttp
import argparse
import subprocess
import sys
from pathlib import Path

INPUT_KEY_MAP = {
    "leval": "input",
    "longbench": "input",
    "loogle": "context",
}


DATASET_CONFIG = {
    "leval": (
        "LEval",
        "LEval-data",
        "/jizhicfs/pkuhetu/sca/data_preprocess/leval_deepseek_results.jsonl",
    ),
    "longbench": (
        "LongBench",
        "Longbench",
        "/jizhicfs/pkuhetu/sca/data_preprocess/longbench_deepseek_results.jsonl",
    ),
    "loogle": (
        "LooGLE",
        "LooGLE_merged",
        "/jizhicfs/pkuhetu/sca/data_preprocess/loogle_deepseek_results.jsonl",
    ),
}
DEFAULT_MODEL = ("/jizhicfs/pkuhetu/sca/models/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659")
DEFAULT_WORKLOAD_ROOT = Path("/jizhicfs/pkuhetu/sca/sca/vllm-dev/examples/test/e2e-exp/reproduce/datasets")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qps", type=float, default=0.02)
    parser.add_argument("--frac", "-f", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    parser.add_argument("--request-timeout-seconds", type=float, default=150 * 60 * 60)
    parser.add_argument("--output-file", "--output_file", "-o", default="output.jsonl")
    parser.add_argument("--model-output-file", default=os.environ.get("MODEL_OUTPUT_FILE"))
    parser.add_argument("--dataset", "-ds", choices=sorted([*DATASET_CONFIG, "tracea"]), default="longbench")
    parser.add_argument("--model-size", "--model_size", "-ms", choices=["8B", "70B"], default="70B")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/jizhicfs/pkuhetu/sca"))
    parser.add_argument("--workload-root", default=os.environ.get("WORKLOAD_ROOT", str(DEFAULT_WORKLOAD_ROOT)))
    parser.add_argument("--api-url", default=os.environ.get("SGLANG_API_URL", "http://127.0.0.1:8055/v1/completions"))
    parser.add_argument("--use-model-output", action="store_true")
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--mapping-config", type=Path)
    parser.add_argument(
        "--tracea-agent",
        type=Path,
        default=Path(
            os.environ.get(
                "TRACEA_AGENT",
                Path(__file__).with_name("replay.py"),
            )
        ),
    )
    parser.add_argument("--summary-file", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-fast-failure-seconds", type=float, default=2.0)
    parser.add_argument("--retry-backoff-seconds", type=float, default=0.5)
    parser.add_argument("--max-in-flight", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--fail-on-request-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_tracea(args):
    if args.input_file is None:
        raise ValueError("--input-file is required for --dataset tracea")
    if not args.input_file.is_file():
        raise FileNotFoundError(f"Missing TraceA input file: {args.input_file}")
    if not args.tracea_agent.is_file():
        raise FileNotFoundError(f"Missing TraceA replay agent: {args.tracea_agent}")

    mapping_config = args.mapping_config or (
        args.input_file.parent / "mapping_config.json"
    )
    if not mapping_config.is_file():
        raise FileNotFoundError(
            f"Missing TraceA mapping config: {mapping_config}"
        )

    command = [
        sys.executable,
        str(args.tracea_agent),
        "--input_file",
        str(args.input_file),
        "--output_file",
        str(args.output_file),
        "--mapping-config",
        str(mapping_config),
        "--model",
        args.model,
        "--api_url",
        args.api_url,
        "--target-qps",
        str(args.qps),
        "--baseline",
        "SGLANG",
        "--model-size",
        args.model_size,
        "--max-attempts",
        str(args.max_attempts),
        "--retry-fast-failure-seconds",
        str(args.retry_fast_failure_seconds),
        "--retry-backoff-seconds",
        str(args.retry_backoff_seconds),
        "--max-in-flight",
        str(args.max_in_flight),
        "--progress-every",
        str(args.progress_every),
    ]
    if args.summary_file is not None:
        command.extend(["--summary-file", str(args.summary_file)])
    if args.run_id:
        command.extend(["--run-id", args.run_id])
    if args.fail_on_request_error:
        command.append("--fail-on-request-error")
    if args.dry_run:
        command.append("--dry-run")

    print("[INFO] TraceA replay command:", " ".join(command))
    subprocess.run(command, check=True)


def resolve_paths(args):
    dataset_prefix, dataset_subdir, default_output_relpath = DATASET_CONFIG[args.dataset]
    data_root = Path(args.data_root)
    workload_root = Path(args.workload_root)
    model_size = args.model_size
    if args.dataset.lower() == "loogle":
        model_size = "70B"  # LooGLE only has 70B model outputs, but consistent with 8B.
    input_file = workload_root / (
        f"{dataset_prefix.lower()}_data/"
        f"outputs_gamma_qps{args.frac}_CV0.25_{model_size}_{args.qps}_subset.csv"
    )
    model_output_file = (
        Path(args.model_output_file)
        if args.model_output_file
        else Path(default_output_relpath)
    )
    return data_root / dataset_subdir, input_file, model_output_file


def load_datasets(dataset_dir):
    datasets = {}
    for jsonl_file in glob.glob(str(dataset_dir / "*.jsonl")):
        file_name = os.path.basename(jsonl_file)
        with open(jsonl_file, "r", encoding="utf-8") as file:
            datasets[file_name] = [json.loads(line) for line in file if line.strip()]
    if not datasets:
        raise FileNotFoundError(f"No JSONL dataset files found under {dataset_dir}")
    return datasets


def load_model_outputs(model_output_file, enabled):
    if not enabled:
        return {}
    model_outputs = {}
    with open(model_output_file, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            key = (item["file"], int(item["line"]), int(item["turn"]))
            model_outputs[key] = item["generate_text"]
    print(f"[INFO] Loaded {len(model_outputs)} model outputs from {model_output_file}")
    return model_outputs


def generate_prompt_and_answer(tokenizer, conversations):
    if not conversations or conversations[-1]["role"] != "assistant":
        raise ValueError("Conversation must end with an assistant standard answer")
    prompt = tokenizer.apply_chat_template(
        conversations[:-1], tokenize=False, add_generation_prompt=True
    )
    prompt_with_answer = tokenizer.apply_chat_template(
        conversations, tokenize=False, add_generation_prompt=False
    )
    encoded_prompt = tokenizer.encode(prompt)
    encoded_prompt_with_answer = tokenizer.encode(prompt_with_answer)
    if encoded_prompt_with_answer[: len(encoded_prompt)] != encoded_prompt:
        raise ValueError("The tokenized answer does not extend the tokenized prompt")
    return prompt, encoded_prompt_with_answer[len(encoded_prompt) :]


def prepare_requests(args, dataframe, datasets, model_outputs, tokenizer):
    for _, row in dataframe.iterrows():
        file_name = str(row["Dataset Name"])
        line_number = int(row["Data Index"])
        turn_id = int(row["Turn"])
        data = datasets[file_name][line_number]
        conversations = [{"role": "user", "content": data[INPUT_KEY_MAP[args.dataset]] + "\n\n"}]
        for idx, question in enumerate(data["instructions"][: turn_id + 1]):
            answer_text = data["outputs"][idx]
            if args.use_model_output:
                answer_text = model_outputs.get((file_name, line_number, idx), answer_text)
            if idx == 0:
                conversations[0]["content"] += question
            else:
                conversations.append({"role": "user", "content": question})
            conversations.append({"role": "assistant", "content": answer_text})
        prompt, answer_token_ids = generate_prompt_and_answer(tokenizer, conversations)
        yield (
            float(row["Timestamp"]),
            prompt,
            answer_token_ids,
            file_name,
            line_number,
            turn_id,
        )


async def write_result(output_file, output_lock, result):
    async with output_lock:
        output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        output_file.flush()


async def fetch(
    session,
    args,
    prompt,
    answer_token_ids,
    output_file,
    output_lock,
    file_name,
    line_number,
    turn_id,
):
    payload = {
        "model": args.model,
        "prompt": prompt,
        "std_output_tokens": answer_token_ids,
        "max_tokens": max(args.max_tokens, len(answer_token_ids)),
        "temperature": 0,
    }
    metadata = {
        "file": file_name,
        "line": line_number,
        "turn": turn_id,
        "std_output_token_count": len(answer_token_ids),
    }
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            async with session.post(args.api_url, json=payload) as response:
                try:
                    response_json = await response.json(content_type=None)
                except Exception:
                    response_json = {"error": {"message": await response.text()}}
                if not isinstance(response_json, dict):
                    response_json = {"response": response_json}
                response_json.update(metadata)
                response_json["http_status"] = response.status
                await write_result(output_file, output_lock, response_json)
                return 200 <= response.status < 300
        except Exception as exc:
            if attempt < max_attempts:
                await asyncio.sleep(0.01)
                continue
            await write_result(
                output_file,
                output_lock,
                {**metadata, "error": {"type": type(exc).__name__, "message": str(exc)}},
            )
            return False


async def schedule_requests(args, dataframe, requests, output_file):
    timeout = aiohttp.ClientTimeout(total=args.request_timeout_seconds)
    connector = aiohttp.TCPConnector(limit=0)
    output_lock = asyncio.Lock()
    tasks = []
    start_time = time.monotonic()

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for request in tqdm.tqdm(requests, total=len(dataframe)):
            timestamp, prompt, answer, file_name, line_number, turn_id = request
            delay = timestamp - (time.monotonic() - start_time)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(
                asyncio.create_task(
                    fetch(
                        session,
                        args,
                        prompt,
                        answer,
                        output_file,
                        output_lock,
                        file_name,
                        line_number,
                        turn_id,
                    )
                )
            )
        succeeded = await asyncio.gather(*tasks)

    failures = len(succeeded) - sum(succeeded)
    if failures:
        raise RuntimeError(f"{failures} request(s) failed; see {args.output_file}")


def main():
    args = parse_args()
    if args.dataset == "tracea":
        run_tracea(args)
        return
    import pandas as pd
    from transformers import AutoTokenizer

    dataset_dir, input_file, model_output_file = resolve_paths(args)
    dataframe = pd.read_csv(input_file)
    datasets = load_datasets(dataset_dir)
    model_outputs = load_model_outputs(model_output_file, args.use_model_output)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    requests = prepare_requests(args, dataframe, datasets, model_outputs, tokenizer)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8", buffering=1) as output_file:
        asyncio.run(schedule_requests(args, dataframe, requests, output_file))


if __name__ == "__main__":
    main()
