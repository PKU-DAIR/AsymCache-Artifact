#!/usr/bin/env bash
# Source this file after activating either experiment environment.
export ARTIFACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VLLM_ROOT="${ARTIFACT_ROOT}/vllm"
export SGLANG_ROOT="${ARTIFACT_ROOT}/sglang-dev"
export ARTIFACT_DATA="${ARTIFACT_DATA:-${ARTIFACT_ROOT}/data}"
export ARTIFACT_RESULTS="${ARTIFACT_RESULTS:-${ARTIFACT_ROOT}/results}"
export VLLM_EXPERIMENT_DATA_ROOT="${ARTIFACT_DATA}/workloads"
export WORKLOAD_ROOT="${VLLM_EXPERIMENT_DATA_ROOT}"
export DATA_ROOT="${WORKLOAD_ROOT}"
export TRACEA_INPUT_FILE="${ARTIFACT_DATA}/workloads/tracea/tracea_2k_cache_pressure_relaxed_2100.jsonl"
export TRACEA_MAPPING_CONFIG="${ARTIFACT_DATA}/workloads/tracea/mapping_config.json"
export TRACEA_DIR="${VLLM_ROOT}/examples/test/e2e-exp/Qwen_trace/auto_script"
export TRACEA_AGENT="${TRACEA_DIR}/replay_traceA.py"
export MODEL_8B_PATH="${MODEL_8B_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
export MODEL_70B_PATH="${MODEL_70B_PATH:-meta-llama/Llama-3.1-70B-Instruct}"
export MODEL_8B="${MODEL_8B_PATH}"
export MODEL_70B="${MODEL_70B_PATH}"
export HF_HOME="${HF_HOME:-${ARTIFACT_DATA}/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export CSET_DIR="${VLLM_ROOT}/csrc/c_set"
export PYTHON_BIN="$(command -v python)"
export PY_VLLM="${PYTHON_BIN}"
export PY_SGLANG="${PYTHON_BIN}"
export RESULTS_ROOT="${ARTIFACT_RESULTS}/raw/vllm"
export DATA_DIR="${ARTIFACT_DATA}/workloads/bfcl"
export VLLM_HOST_IP=127.0.0.1
export VLLM_SIMPLE_SHED_ALGO=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export NCCL_ALGO=Ring
export flash_attn_src="${ARTIFACT_ROOT}/vllm-flash-attn-dev"
# This is the variable consumed by vLLM's CMake integration.
export VLLM_FLASH_ATTN_SRC_DIR="${flash_attn_src}"
export USE_ORG=1 USE_POD=1
export PYTHONPATH="${CSET_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
