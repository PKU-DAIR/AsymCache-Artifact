#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${REPRO_ROOT}/env.sh"

COMMON_ENV=(
  RESULTS_BASE_DIR="${RESULTS_ROOT}/loogle/70B"
  MODEL="${MODEL_70B}"
  BS_LIST="${BS_LIST_70B}"
  TP="${TP_70B}"
  GPU="${GPU_70B}"
  PORT="${PORT_70B}"
)

env "${COMMON_ENV[@]}" \
  VLLM_PERFORMANCE_ALIGNMENT=0 \
  CONFIG_FILE="${SCRIPT_DIR}/config_qps_0.02_0.04.txt" \
  bash "${SCRIPT_DIR}/run_sweep_qps_0.02_0.04.sh"

env "${COMMON_ENV[@]}" \
  VLLM_PERFORMANCE_ALIGNMENT=1 \
  CONFIG_FILE="${SCRIPT_DIR}/config_qps_0.01.txt" \
  bash "${SCRIPT_DIR}/run_sweep_qps_0.01.sh"
