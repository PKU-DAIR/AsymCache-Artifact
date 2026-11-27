#!/usr/bin/env bash
set -uo pipefail

REPRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${REPRO_ENV_FILE:-${REPRO_ROOT}/env.sh}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[ERROR] Missing environment file: ${ENV_FILE}" >&2
  exit 1
fi
source "${ENV_FILE}"

DATASETS="${DATASETS:-longbench loogle traceA}"
MODEL_SIZES="${MODEL_SIZES:-8B 70B}"

contains_word() {
  local needle="$1"
  local value
  for value in $2; do
    [[ "${value}" == "${needle}" ]] && return 0
  done
  return 1
}

missing_inputs=()
require_file() {
  [[ -f "$1" ]] || missing_inputs+=("file: $1")
}
require_dir() {
  [[ -d "$1" ]] || missing_inputs+=("directory: $1")
}

require_dir "${SGLANG_ROOT}/python/sglang"
contains_word 8B "${MODEL_SIZES}" && require_dir "${MODEL_8B}"
contains_word 70B "${MODEL_SIZES}" && require_dir "${MODEL_70B}"

if contains_word longbench "${DATASETS}"; then
  require_dir "${DATA_ROOT}/Longbench"
  require_file "${MODEL_OUTPUT_FILE:-${DATA_ROOT}/data_preprocess/longbench_deepseek_results.jsonl}"
  for model_size in 8B 70B; do
    contains_word "${model_size}" "${MODEL_SIZES}" || continue
    for qps in 0.01 0.02 0.04; do
      for frac in 5 10; do
        require_file "${WORKLOAD_ROOT}/longbench_data/outputs_gamma_qps${frac}_CV0.25_${model_size}_${qps}_subset.csv"
      done
    done
  done
fi

if contains_word loogle "${DATASETS}"; then
  require_dir "${DATA_ROOT}/LooGLE_merged"
  require_file "${MODEL_OUTPUT_FILE:-${DATA_ROOT}/data_preprocess/loogle_deepseek_results.jsonl}"
  for qps in 0.01 0.02 0.04; do
    for frac in 5 10; do
      require_file "${WORKLOAD_ROOT}/loogle_data/outputs_gamma_qps${frac}_CV0.25_70B_${qps}_subset.csv"
    done
  done
fi

if contains_word traceA "${DATASETS}"; then
  require_file "${TRACEA_INPUT_FILE}"
  require_file "${TRACEA_MAPPING_CONFIG}"
  require_file "${TRACEA_AGENT}"
fi

if (( ${#missing_inputs[@]} > 0 )); then
  echo "[ERROR] Reproduction prerequisites are incomplete:" >&2
  printf '  - %s\n' "${missing_inputs[@]}" >&2
  exit 1
fi

if ! mkdir -p \
    "${RESULTS_ROOT}/longbench/8B" "${RESULTS_ROOT}/longbench/70B" \
    "${RESULTS_ROOT}/loogle/8B" "${RESULTS_ROOT}/loogle/70B" \
    "${RESULTS_ROOT}/traceA/8B" "${RESULTS_ROOT}/traceA/70B"; then
  echo "[ERROR] Cannot create results tree under ${RESULTS_ROOT}" >&2
  exit 1
fi

failures=()
run_stage() {
  local dataset="$1"
  local model_size="$2"
  local script="$3"

  if ! contains_word "${dataset}" "${DATASETS}" || \
     ! contains_word "${model_size}" "${MODEL_SIZES}"; then
    echo "[SKIP] ${dataset}/${model_size}"
    return 0
  fi

  echo "============================================================"
  echo "[RUN] ${dataset}/${model_size}"
  echo "============================================================"
  if bash "${script}"; then
    echo "[DONE] ${dataset}/${model_size}"
  else
    echo "[FAILED] ${dataset}/${model_size}" >&2
    failures+=("${dataset}/${model_size}")
  fi
}

run_stage longbench 8B "${REPRO_ROOT}/scripts/longbench/8B/run.sh"
run_stage longbench 70B "${REPRO_ROOT}/scripts/longbench/70B/run.sh"
run_stage loogle 8B "${REPRO_ROOT}/scripts/loogle/8B/run.sh"
run_stage loogle 70B "${REPRO_ROOT}/scripts/loogle/70B/run.sh"
run_stage traceA 8B "${REPRO_ROOT}/scripts/traceA/8B/run.sh"
run_stage traceA 70B "${REPRO_ROOT}/scripts/traceA/70B/run.sh"

if (( ${#failures[@]} > 0 )); then
  echo "[SUMMARY] Failed stages: ${failures[*]}" >&2
  exit 1
fi

echo "[SUMMARY] All requested experiments completed successfully."
echo "[SUMMARY] Results root: ${RESULTS_ROOT}"
