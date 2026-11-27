#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
target="${1:?Usage: bash scripts/run.sh e2e|tracea|agent|sensitivity|sglang [options]}"
shift
case "${target}" in
  e2e) exec python "${ARTIFACT_ROOT}/scripts/run_e2e.py" "$@" ;;
  tracea)
    exec python "${ARTIFACT_ROOT}/scripts/run_tracea.py" "$@"
    ;;
  agent)
    for size in ${MODEL_SIZES:-8B 70B}; do
      for qps in 1.5 1.75 2.0; do
        for baseline in LRU MWF CONTINUUM CONTINUUM_MWF; do
          python "${ARTIFACT_ROOT}/scripts/archive_outputs.py" \
            "${RESULTS_ROOT}/agent/${size}/2x_${baseline}_qps${qps}"
          bash "${VLLM_ROOT}/examples/test/e2e-exp/BFCL_workload_reproduce/run_bfcl_baseline_server.sh" \
            --baseline "${baseline}" --model-size "${size}" --qps "${qps}" \
            --result-root "${RESULTS_ROOT}/agent/${size}" "$@"
        done
      done
    done
    ;;
  sensitivity)
    export BASE_DIR="${RESULTS_ROOT}/sensitivity"
    python "${ARTIFACT_ROOT}/scripts/archive_outputs.py" "${BASE_DIR}"
    bash "${VLLM_ROOT}/examples/test/e2e-exp/hyper_parameter/run_ab_sweep.sh" "$@"
    # The default point and LRU reference are collected with the same replay.
    for baseline in LRU ASYMCACHE; do
      python "${ARTIFACT_ROOT}/scripts/run_e2e.py" --baseline "${baseline}" \
        --workload longbench --model-size 8B --qps 0.02
    done
    ;;
  sglang)
    export RESULTS_ROOT="${ARTIFACT_RESULTS}/raw/sglang"
    export TRACEA_AGENT="${SGLANG_ROOT}/workload-exp/exp-reproduce/scripts/traceA/replay.py"
    exec bash "${SGLANG_ROOT}/workload-exp/exp-reproduce/run_all.sh" "$@"
    ;;
  *) echo "Unknown experiment group: ${target}" >&2; exit 2 ;;
esac
