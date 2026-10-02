#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPRO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/jizhicfs/pkuhetu/sca/models/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export USE_POD=1
export USE_ORG=1

PY_SGLANG="${PY_SGLANG:-python3}"
DATA_ROOT="${DATA_ROOT:-/jizhicfs/pkuhetu/sca}"
WORKLOAD_ROOT="${WORKLOAD_ROOT:-${REPRO_ROOT}/datasets}"
SGLANG_ROOT="${SGLANG_ROOT:-/jizhicfs/pkuhetu/sca/sca/sgl-pod}"
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-${REPRO_ROOT}/results/loogle/8B}"
MODEL="${MODEL:-${MODEL_8B:-/jizhicfs/pkuhetu/sca/models/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}}"
EXP_INTERFACE="${EXP_INTERFACE:-${DATASET_ROOT}/interface.py}"

TP="${TP:-1}"
GPU="${GPU:-7}"
PORT="${PORT:-8050}"
DTYPE="${DTYPE:-float16}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-450000}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"
SERVER_HOST="${SGLANG_SERVER_HOST:-127.0.0.1}"
SCHEDULE_POLICY="${SGLANG_SCHEDULE_POLICY:-fcfs}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
BS_LIST="${BS_LIST:-2 4 $(seq 8 32) $(seq 48 16 160)}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-487744}"
VLLM_PERFORMANCE_ALIGNMENT="${VLLM_PERFORMANCE_ALIGNMENT:-0}"

P=""
QPS=""
GRAD=""
MINEXP_B=""
MINEXP_ALPHA=""
MINEXP_BETA=""
FRAC=""
DATASET=""
MODEL_SIZE="8B"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qps)          QPS="$2"; shift 2 ;;
    --p)            P="$2"; shift 2 ;;
    --grad)         GRAD="$2"; shift 2 ;;
    --minexp-b)     MINEXP_B="$2"; shift 2 ;;
    --minexp-alpha) MINEXP_ALPHA="$2"; shift 2 ;;
    --minexp-beta)  MINEXP_BETA="$2"; shift 2 ;;
    --dataset)      DATASET="$2"; shift 2 ;;
    --model-size)   MODEL_SIZE="$2"; shift 2 ;;
    --frac)         FRAC="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

for required in QPS P GRAD MINEXP_B MINEXP_ALPHA MINEXP_BETA DATASET FRAC; do
  if [[ -z "${!required}" ]]; then
    echo "Missing required argument for ${required}" >&2
    exit 1
  fi
done

case "${DATASET}" in
  leval|longbench|loogle) ;;
  *) echo "Unsupported dataset: ${DATASET}" >&2; exit 1 ;;
esac

if [[ ! -d "${SGLANG_ROOT}/python/sglang" ]]; then
  echo "SGLANG_ROOT does not contain python/sglang: ${SGLANG_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${EXP_INTERFACE}" ]]; then
  echo "Missing experiment interface: ${EXP_INTERFACE}" >&2
  exit 1
fi

# These variables keep the existing sweep-file contract. Stock SGLang does not
# consume the Pensieve/MinExp knobs unless a custom scheduler implementation does.
export P GRAD MINEXP_B MINEXP_ALPHA MINEXP_BETA
export PYTHONPATH="${SGLANG_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

EXP_NAME="300_${DATASET}_sglang_${SCHEDULE_POLICY}_${MODEL_SIZE}_overall_qps${QPS}_frac${FRAC}_CV0.25_gamma"
EXP_DIR="${RESULTS_BASE_DIR}/${EXP_NAME}"
mkdir -p "${EXP_DIR}"

SCHED_LOG="${EXP_DIR}/scheduler.log"
SERVER_LOG="${EXP_DIR}/server_output.log"
OUTPUT_JSONL="${EXP_DIR}/${EXP_NAME}.jsonl"
API_URL="http://${SERVER_HOST}:${PORT}/v1/completions"

export SCHEDULE_LOG_PATH="${SCHED_LOG}"

wait_for_server() {
  echo "Waiting up to ${STARTUP_TIMEOUT}s for SGLang on ${SERVER_HOST}:${PORT}..."
  local start_time=$SECONDS
  while (( SECONDS - start_time < STARTUP_TIMEOUT )); do
    if curl -fsS "http://${SERVER_HOST}:${PORT}/health_generate" >/dev/null 2>&1; then
      echo "[READY] SGLang server is ready."
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[ERROR] SGLang server exited during startup."
      echo "========== server_output.log =========="
      tail -n 100 "${SERVER_LOG}"
      echo "======================================="
      return 1
    fi
    sleep 1
  done
  echo "[ERROR] SGLang startup timed out after ${STARTUP_TIMEOUT}s."
  tail -n 100 "${SERVER_LOG}"
  return 1
}

flush_cache_when_idle() {
  local flush_timeout=30
  local start_time=$SECONDS
  local response_file="${EXP_DIR}/flush_cache_response.txt"

  while (( SECONDS - start_time < flush_timeout )); do
    if curl --fail --silent --show-error \
      -X POST "http://${SERVER_HOST}:${PORT}/flush_cache" \
      >"${response_file}" 2>/dev/null; then
      rm -f "${response_file}"
      echo "[READY] SGLang cache flushed."
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[ERROR] SGLang server exited while flushing cache."
      tail -n 100 "${SERVER_LOG}"
      return 1
    fi
    sleep 1
  done

  echo "[ERROR] Cache remained busy for ${flush_timeout}s."
  cat "${response_file}" 2>/dev/null || true
  return 1
}

CUDA_VISIBLE_DEVICES="${GPU}" setsid "${PY_SGLANG}" \
  -m sglang.launch_server \
  --model-path "${MODEL}" \
  --host "${SERVER_HOST}" \
  --port "${PORT}" \
  --tp-size "${TP}" \
  --dtype "${DTYPE}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --context-length "${CONTEXT_LENGTH}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --schedule-policy "${SCHEDULE_POLICY}" \
  --log-level debug \
  --max-total-tokens "${MAX_TOTAL_TOKENS}" \
  --attention-backend fapod \
  --prefill-attention-backend fapod \
  --decode-attention-backend fapod \
  --page-size 64 \
  --enable-mixed-chunk \
  --enable-cache-report \
  --disable-overlap-schedule \
  --cuda-graph-bs ${BS_LIST} \
  --enable-vllm-performance-alignment "${VLLM_PERFORMANCE_ALIGNMENT}" \
  >"${SERVER_LOG}" 2>&1 &

SERVER_PID=$!
SERVER_PGID="$(ps -o pgid= "${SERVER_PID}" | tr -d ' ' || true)"

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "${SERVER_PGID:-}" ]]; then
    echo "[CLEANUP] Stopping SGLang process group ${SERVER_PGID}"
    kill -TERM -- "-${SERVER_PGID}" 2>/dev/null
    sleep 6
    kill -KILL -- "-${SERVER_PGID}" 2>/dev/null
  elif [[ -n "${SERVER_PID:-}" ]]; then
    kill -TERM "${SERVER_PID}" 2>/dev/null
  fi
  return "${exit_code}"
}

handle_interrupt() {
  echo "[INTERRUPT] Received Ctrl+C, exiting..."
  exit 130
}

handle_terminate() {
  echo "[TERMINATE] Received SIGTERM, exiting..."
  exit 143
}

trap cleanup EXIT
trap handle_interrupt INT
trap handle_terminate TERM

wait_for_server
# health_generate warms a one-token request; clear it before collecting results.
flush_cache_when_idle

client_args=(
  "${EXP_INTERFACE}"
  --qps "${QPS}"
  --frac "${FRAC}"
  --dataset "${DATASET}"
  --model-size "${MODEL_SIZE}"
  --model "${MODEL}"
  --data-root "${DATA_ROOT}"
  --workload-root "${WORKLOAD_ROOT}"
  --api-url "${API_URL}"
  --output-file "${OUTPUT_JSONL}"
  --use-model-output
)

"${PY_SGLANG}" "${client_args[@]}"
echo "[DONE] Experiment finished in ${EXP_DIR}"
