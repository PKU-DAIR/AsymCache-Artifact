#!/usr/bin/env bash

REPRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SGLANG_ROOT="${SGLANG_ROOT:-/jizhicfs/pkuhetu/sca/sca/sgl-pod}"
export DATA_ROOT="${DATA_ROOT:-/jizhicfs/pkuhetu/sca}"
export WORKLOAD_ROOT="${WORKLOAD_ROOT:-/jizhicfs/pkuhetu/sca/sca/vllm-dev/examples/test/e2e-exp/reproduce/datasets}"
export RESULTS_ROOT="${RESULTS_ROOT:-${REPRO_ROOT}/results}"

export MODEL_8B="${MODEL_8B:-/jizhicfs/pkuhetu/sca/models/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"
export MODEL_70B="${MODEL_70B:-/jizhicfs/pkuhetu/sca/models/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b}"
export BS_LIST_8B="${BS_LIST_8B:-2 4 $(seq 8 32) $(seq 48 16 160)}"
export BS_LIST_70B="${BS_LIST_70B:-$(seq 8 32) $(seq 48 16 160)}"

export TP_8B="${TP_8B:-1}"
export GPU_8B="${GPU_8B:-7}"
export PORT_8B="${PORT_8B:-8060}"
export TP_70B="${TP_70B:-4}"
export PORT_70B="${PORT_70B:-8050}"
export GPU_70B="${GPU_70B:-4,5,6,7}"

export TRACEA_INPUT_FILE="${TRACEA_INPUT_FILE:-/jizhicfs/pkuhetu/sca/sca/sgl-pod/workload-exp/tracea_2k_cache_pressure_relaxed_2100.jsonl}"
export TRACEA_MAPPING_CONFIG="${TRACEA_MAPPING_CONFIG:-/jizhicfs/pkuhetu/sca/sca/sgl-pod/workload-exp/mapping_config.json}"
export TRACEA_AGENT="${TRACEA_AGENT:-${REPRO_ROOT}/scripts/traceA/replay.py}"

export SWEEP_COOLDOWN_SECONDS="${SWEEP_COOLDOWN_SECONDS:-60}"
