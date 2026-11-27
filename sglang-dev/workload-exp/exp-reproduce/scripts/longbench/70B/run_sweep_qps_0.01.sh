#!/usr/bin/env bash
set -Eeuo pipefail

TP="${TP:-4}"
GPU="${GPU:-4,5,6,7}"
PORT="${PORT:-8050}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/config_qps_0.01.txt}"
COOLDOWN_SECONDS="${SWEEP_COOLDOWN_SECONDS:-60}"

while read -r qps p grad minexp_b minexp_alpha minexp_beta dataset model_size frac; do
  [[ -z "${qps:-}" || "${qps:0:1}" == "#" ]] && continue

  echo "[SWEEP] qps=${qps}, dataset=${dataset}, model_size=${model_size}, frac=${frac}"
  TP="${TP}" GPU="${GPU}" PORT="${PORT}" bash "${SCRIPT_DIR}/run_one_qps_0.01.sh" \
      --qps "${qps}" \
      --p "${p}" \
      --grad "${grad}" \
      --minexp-b "${minexp_b}" \
      --minexp-alpha "${minexp_alpha}" \
      --minexp-beta "${minexp_beta}" \
      --dataset "${dataset}" \
      --model-size "${model_size}" \
      --frac "${frac}"

  sleep "${COOLDOWN_SECONDS}"
  echo "[SWEEP] Finished qps=${qps}, dataset=${dataset}, model_size=${model_size}, frac=${frac}"
done < "${CONFIG_FILE}"
