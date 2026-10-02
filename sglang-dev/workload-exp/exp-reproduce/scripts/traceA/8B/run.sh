#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${REPRO_ROOT}/env.sh"

CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/config.txt}" \
RESULTS_BASE_DIR="${RESULTS_ROOT}/traceA/8B" \
MODEL="${MODEL_8B}" BS_LIST="${BS_LIST_8B}" VLLM_PERFORMANCE_ALIGNMENT=0 \
TP="${TP_8B}" GPU="${GPU_8B}" PORT="${PORT_8B}" \
bash "${SCRIPT_DIR}/run_sweep.sh"
