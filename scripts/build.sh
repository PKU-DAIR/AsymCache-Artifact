#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
target="${1:?Usage: bash scripts/build.sh vllm|sglang}"
export MAX_JOBS="${MAX_JOBS:-16}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
python -m pip install 'setuptools>=77.0.3,<80' wheel packaging ninja 'cmake>=3.26' pybind11
case "${target}" in
  vllm)
    test -f "${flash_attn_src}/CMakeLists.txt"
    python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    python -m pip install -r "${VLLM_ROOT}/requirements/build.txt"
    python -m pip install -r "${VLLM_ROOT}/requirements/cuda.txt"
    (cd "${VLLM_ROOT}" && python -m pip install -e . --no-build-isolation -v)
    ;;
  sglang)
    python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
    python -m pip install -e "${SGLANG_ROOT}/python[srt,openai]" --extra-index-url https://flashinfer.ai/whl/cu128/torch2.7/
    python "${ARTIFACT_ROOT}/scripts/link_flash_attn.py"
    ;;
  *) echo "Unknown build target: ${target}" >&2; exit 2 ;;
esac
# Keep the two POD builds in separate source trees.
if [[ "${target}" == vllm ]]; then
  (cd "${ARTIFACT_ROOT}/pod_attn_dev/pod_attn" && python -m pip install -e . --no-build-isolation -v)
else
  python "${ARTIFACT_ROOT}/scripts/build_sglang_pod.py"
fi
python -m pip install -r "${ARTIFACT_ROOT}/scripts/requirements.txt"
(cd "${CSET_DIR}" && c++ -O3 -Wall -shared -std=c++17 -fPIC \
  $(python -m pybind11 --includes) c_set.cpp \
  -o "c_set$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')")
python "${ARTIFACT_ROOT}/scripts/check_build.py" "${target}"
