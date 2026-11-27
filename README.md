# AsymCache Artifact

This artifact reproduces AsymCache's end-to-end serving results, its evaluation on the Qwen-Bailian production workload, the Multi-Segment Attention (MSA) kernel comparison, the hyper-parameter sensitivity study, and its integration with Continuum on an agent workload.

The [main repository](https://github.com/PKU-DAIR/AsymCache-Artifact.git) contains the experiment tools, workload inputs, paper reference measurements, plotting code, customized SGLang source, and POD attention source. The [customized vLLM repository](https://github.com/SpiritedAwayCN/AsymCache-Artifact-vLLM.git) is checked out at `vllm/`. All vLLM experiments use this unified checkout.

The proof for the order-preserving rule in Section 4.4 is available in [the supplementary proof](%5BPROOF%5D%20properties%20of%20order-preserving%20rule.pdf).

## 1. Paper results and terminology

| Paper result | Workload and experiment | Run command | Figure basename |
| --- | --- | --- | --- |
| Figure 11 | LongBench and LooGLE, low dispersion (`frac=5`) | `bash scripts/run.sh e2e` and `bash scripts/run.sh sglang` | `figure_e2e_f5` |
| Figure 12 | LongBench and LooGLE, high dispersion (`frac=10`) | Same end-to-end runs | `figure_e2e_f10` |
| Figure 13 | Qwen-Bailian real workload, named **TraceA** in the code | `bash scripts/run.sh tracea` and `bash scripts/run.sh sglang` | `figure_tracea` |
| Figure 14 | MSA kernel comparison on Llama 3.1-8B | `python scripts/benchmark_kernel.py` | `ablation-fused2` |
| Figure 15 | Hyper-parameter sensitivity on LongBench, Llama 3.1-8B | `bash scripts/run.sh sensitivity` | `sense` |
| Figure 16 | Agent workload from BFCL v4 Web Search; orthogonality with Continuum | `bash scripts/run.sh agent` | `figure_agent` |

LongBench and LooGLE are the paper's end-to-end workloads. TraceA denotes the replay input for the Qwen-Bailian dataset; it is evaluated using Llama 3.1-8B and Llama 3.1-70B. The BFCL workload is the paper's agent workload. `MWF` is the implementation name of AsymCache's block eviction policy, and `HOTPREFIX` is the implementation name of the Min-Hotness baseline.

The plots report TTFT in seconds, TPOT in milliseconds, and cache hit rates in percent. The replay summaries store latency values in seconds and hit rates as fractions. TPOT follows the paper's analysis scripts: request completion time from arrival, divided by the number of output tokens. Agent plots report average and P90 job completion times, measured from the first arrival to the final completion within each `sample_id`.

## 2. Repository layout

```text
AsymCache-Artifact/
  vllm/                     # Unified vLLM source submodule
  vllm-flash-attn-dev/      # FlashAttention source submodule
  sglang-dev/               # Customized SGLang source
  pod_attn_dev/pod_attn/     # POD/MSA extension source
  scripts/                  # Environment, build, workload and validation tools
  plots/                    # Replay analysis and paper figure scripts
  data/
    workloads/              # Conversation inputs, arrival schedules and traces
    reference/paper/        # Paper measurements, including kernel profiles
    hf-cache/               # Model download cache
  results/
    raw/vllm/               # End-to-end, TraceA, agent and sensitivity outputs
    raw/sglang/             # SGLang replay outputs
    raw/kernel/             # Kernel profiling CSV files
    parameters/             # TraceA MWF parameters, one JSON file per QPS
    summary/                # Merged CSV tables for plotting
    smoke/                  # Synthetic workflow validation outputs
  figures/                  # Generated PDF and PNG figures
```

See [data/README.md](data/README.md) for input formats and [plots/README.md](plots/README.md) for individual plotting commands.

## 3. Clone and configure

The experiment configuration uses Linux x86-64, eight NVIDIA H20 GPUs, CUDA Toolkit 12.8, and Python 3.10. The 8B model uses one GPU; the 70B model uses four GPUs with tensor parallelism. The two serving environments use the PyTorch versions specified by their source packages: 2.7.0 for vLLM and 2.7.1 for SGLang, with CUDA 12.8 wheels.

```bash
git clone --recurse-submodules https://github.com/PKU-DAIR/AsymCache-Artifact.git
cd AsymCache-Artifact
export ARTIFACT_ROOT="$PWD"
```

For an existing clone:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

The recorded submodule revisions are:

| Directory | Source | Revision |
| --- | --- | --- |
| `vllm` | `https://github.com/SpiritedAwayCN/AsymCache-Artifact-vLLM.git` | `eb1f068d5fd78fb68a861a751caa5d9dc3e71904` (`main`) |
| `vllm-flash-attn-dev` | `https://github.com/SpiritedAwayCN/vllm-flash-attenntion-dev.git` | `699ab0c98e058f40ab46cc148c046bd8f356657e` (`250512-rebase`, default branch) |

Install the system build tools:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git curl ninja-build numactl
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export MAX_JOBS=16
export TORCH_CUDA_ARCH_LIST=9.0
```

Set the local model directories before sourcing the shared environment:

```bash
export MODEL_8B_PATH=/path/to/Llama-3.1-8B-Instruct
export MODEL_70B_PATH=/path/to/Llama-3.1-70B-Instruct
source scripts/env.sh
```

The model directories contain the model weights, tokenizer files and model configuration. `scripts/env.sh` derives the code, workload, extension and output paths from the checkout location. Source it again after activating an environment so that `PYTHON_BIN`, `PY_VLLM` and `PY_SGLANG` select that environment's interpreter. Set `ARTIFACT_DATA` and `ARTIFACT_RESULTS` before sourcing it to place data and results on another filesystem.

The shared environment enables the engines' context-length override settings for the experiment scripts' 450,000-token capacity.

## 4. Build the two environments

### 4.1 vLLM environment

```bash
conda create -n asymcache-vllm python=3.10 -y
conda activate asymcache-vllm
cd "$ARTIFACT_ROOT"
source scripts/env.sh
bash scripts/build.sh vllm
```

The build script installs vLLM's build and CUDA dependencies, compiles the local vLLM source in editable mode, builds POD attention, and compiles `vllm/csrc/c_set/c_set.cpp`.

The local FlashAttention source is selected with both environment variables below. vLLM's CMake integration reads `VLLM_FLASH_ATTN_SRC_DIR`:

```bash
export flash_attn_src="${ARTIFACT_ROOT}/vllm-flash-attn-dev"
export VLLM_FLASH_ATTN_SRC_DIR="${flash_attn_src}"
```

The source build commands executed by the script are:

```bash
cd "${ARTIFACT_ROOT}/vllm"
python -m pip install -e . --no-build-isolation -v

cd "${ARTIFACT_ROOT}/pod_attn_dev/pod_attn"
python -m pip install -e . --no-build-isolation -v

cd "${ARTIFACT_ROOT}/vllm/csrc/c_set"
c++ -O3 -Wall -shared -std=c++17 -fPIC \
  $(python -m pybind11 --includes) c_set.cpp \
  -o "c_set$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
```

### 4.2 SGLang environment

```bash
conda create -n asymcache-sglang python=3.10 -y
conda activate asymcache-sglang
cd "$ARTIFACT_ROOT"
source scripts/env.sh
bash scripts/build.sh sglang
```

This installs the customized SGLang source with `pip install -e ./python[srt,openai]`, builds POD attention in the SGLang environment, and compiles `c_set`.

The SGLang POD build uses a source copy at `<sglang-environment>/src/asymcache-pod-attn/` and installs it in editable mode. The vLLM POD build uses `pod_attn_dev/pod_attn/`. This gives each environment its own POD binary compiled against its PyTorch version.

The build script creates this symbolic link:

```text
sglang-dev/python/sglang/vllm_flash_attn_org
  -> <artifact-root>/vllm/vllm/vllm_flash_attn
```

The SGLang backend runs with `USE_ORG=1` and `USE_POD=1`. In `fapod_backend.py`, `USE_ORG=1` selects `_vllm_fa2_C*.so` through `vllm_flash_attn_org`; `USE_POD=1` imports the POD extension. Both serving engines use the FlashAttention library compiled by the vLLM build. Recreate the link after relocating the checkout with:

```bash
python scripts/link_flash_attn.py
```

Each build ends by importing the local serving package, POD attention and `c_set`. The SGLang check also loads the shared FA2 library through the actual attention backend. Run these checks again with:

```bash
conda activate asymcache-vllm
source scripts/env.sh
python scripts/check_build.py vllm

conda activate asymcache-sglang
source scripts/env.sh
python scripts/check_build.py sglang
```

## 5. Prepare workloads

All experiment inputs are included under `data/workloads/`: LongBench and LooGLE conversations, their generated reference answers and arrival schedules, the BFCL CSV files, and the 2,100-request TraceA input with its token mapping. Validate the packaged inputs with:

```bash
conda activate asymcache-vllm
cd "$ARTIFACT_ROOT"
source scripts/env.sh
python scripts/prepare_data.py
```

LongBench contains 380 conversations in six JSONL files and 1,140 generated reference answers. LooGLE contains 140 conversations in `longdep_qa.jsonl` and 1,101 generated reference answers. The reference files are `data_preprocess/longbench_deepseek_results.jsonl` and `data_preprocess/loogle_deepseek_results.jsonl`. The replay interfaces use these answers to construct multi-turn histories and standard output tokens.

The validator checks conversation schemas, chronological arrival schedules, dialogue and turn indices, unique reference keys, and reference answers for every replayed turn. Select experiment groups with:

```bash
python scripts/prepare_data.py --groups e2e sensitivity
python scripts/prepare_data.py --groups tracea agent
```

## 6. Run experiments

Run the following commands from the artifact root. The launchers write replay records, server logs and scheduler logs into the common `results/raw/` tree.

### 6.1 LongBench and LooGLE: end-to-end experiments

```bash
conda activate asymcache-vllm
source scripts/env.sh
bash scripts/run.sh e2e
```

The launcher reads the workload configurations in `vllm/examples/test/e2e-exp/reproduce/e2e/lru/`. It runs LongBench and LooGLE, both model sizes, QPS 0.01/0.02/0.04, and dispersion factors 5/10. The matrix contains 124 vLLM configurations.

| Launcher baseline | Figure label | Eviction policy | `USE_MINEXP` |
| --- | --- | --- | --- |
| `LRU` | vLLM-LRU | `lru` | 0 |
| `PENSIEVE` | Pensieve + MSA | `pensieve` | 0 |
| `BS` | Max-Score | `bs` | 2 |
| `HOTPREFIX` | Min-Hotness | `hotprefix` | 0 |
| `ASYMCACHE_NO_SCHEDULER` | AsymCache w/o Scheduler | `mwf` | 2 |
| `ASYMCACHE` | AsymCache | `mwf` | 2 |

Select an individual configuration with:

```bash
bash scripts/run.sh e2e --baseline BS --workload longbench \
  --model-size 8B --qps 0.02 --frac 5 --gpu 0 --port 8050
```

Print the complete matrix with `bash scripts/run.sh e2e --dry-run`. Completed configurations are recorded in `run.json` and skipped when resuming; add `--rerun` to execute them again. Outputs are organized as:

```text
results/raw/vllm/e2e/<baseline>/<experiment>/
  <experiment>.jsonl
  run.json
  server_output.log
  scheduler.log
```

### 6.2 TraceA: Qwen-Bailian real workload

```bash
bash scripts/run.sh tracea
```

This runs LRU, Pensieve + MSA, Max-Score, AsymCache and Min-Hotness on both model sizes at QPS 0.2, 0.25 and 0.3: 30 vLLM configurations. For each QPS, the launcher invokes `Qwen_trace/CDF.py` at percentile 75, reads the generated JSON parameters and passes `MINEXP_B`, `MINEXP_ALPHA` and `MINEXP_BETA` to the experiment launcher. Min-Hotness uses aging interval 2.

```bash
bash scripts/run.sh tracea --model-size 8B --qps 0.2 --baseline MWF
bash scripts/run.sh tracea --model-size 70B --qps 0.25 --dry-run
```

Parameters are stored in `results/parameters/tracea_qps<QPS>.json`; replay outputs are under `results/raw/vllm/tracea/<8B|70B>/<experiment>/`.

### 6.3 SGLang baseline for end-to-end and TraceA

```bash
conda activate asymcache-sglang
source scripts/env.sh
bash scripts/run.sh sglang
```

The six stages run LongBench, LooGLE and TraceA on both model sizes. They produce 24 end-to-end configurations and six TraceA configurations under `results/raw/sglang/<longbench|loogle|traceA>/<8B|70B>/`.

Select stages and GPU placement through environment variables:

```bash
DATASETS="longbench loogle" MODEL_SIZES="8B" GPU_8B=0 \
  bash scripts/run.sh sglang
DATASETS="traceA" MODEL_SIZES="70B" GPU_70B=0,1,2,3 \
  bash scripts/run.sh sglang
```

### 6.4 BFCL: agent workload and Continuum orthogonality

```bash
conda activate asymcache-vllm
source scripts/env.sh
bash scripts/run.sh agent
```

The matrix contains both model sizes, QPS 1.5/1.75/2.0, and four baselines: `LRU`, `MWF`, `CONTINUUM`, and `CONTINUUM_MWF`, for 24 runs. They correspond to vLLM-LRU, AsymCache, Continuum, and Continuum + AsymCache. Outputs are under `results/raw/vllm/agent/<8B|70B>/2x_<BASELINE>_qps<QPS>/`.

Run one configuration directly with the same paths:

```bash
bash "${VLLM_ROOT}/examples/test/e2e-exp/BFCL_workload_reproduce/run_bfcl_baseline_server.sh" \
  --baseline CONTINUUM_MWF --model-size 70B --qps 1.75 \
  --result-root "${RESULTS_ROOT}/agent/70B"
```

### 6.5 Hyper-parameter sensitivity

```bash
bash scripts/run.sh sensitivity
```

The 20 configurations in `hyper_parameter/ablation_config.txt` sweep lifespan multipliers 0.6/0.8/1.2/1.4, reuse probabilities 0.80/0.90/0.95, and slope change ratios 10/25/50. Each sweep evaluates both dispersion factors on LongBench, Llama 3.1-8B, QPS 0.02. The launcher also collects the default AsymCache point and LRU reference for both factors using the end-to-end launcher. The plotting code inserts these references into all six panels automatically.

Sweep results are under `results/raw/vllm/sensitivity/`. The default points and LRU references are under `results/raw/vllm/e2e/`.

### 6.6 MSA kernel comparison

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_kernel.py
```

The profiler runs the three caching methods in separate processes for 1K and 10K sequences, using Llama 3.1-8B and tensor parallelism 1. It varies cached length in 16-token steps and records per-run model forward times and the trimmed mean in milliseconds. The suffix experiments use 32 appended tokens for the 1K profile and 128 for the 10K profile, following the original profiling scripts. `--tail-tokens` selects a common appended-token count.

```bash
python scripts/benchmark_kernel.py --dry-run
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_kernel.py --mode msa --length 10k
```

The six CSV files in `results/raw/kernel/` use the original `cached_length` and `AVG` columns and filenames.

## 7. Summarize and draw the figures

After collecting the runs in both environments:

```bash
cd "$ARTIFACT_ROOT"
python plots/collect_results.py
python plots/plot_all.py
```

The collector uses each experiment family's analyzer and merges vLLM and SGLang results into four named tables: `e2e.csv`, `tracea.csv`, `agent.csv`, and `sensitivity.csv` in `results/summary/`. The plotter reads those column names, assembles the sensitivity panels and writes all six figures as PDF and PNG under `figures/`. It preserves the paper's subplot layouts, Calibri typography, color palettes, hatching, axes and figure basenames. Install Calibri in the plotting environment to use the paper's font.

To draw the figures directly from the included paper measurements:

```bash
python -m pip install -r scripts/requirements.txt
python plots/plot_all.py --reference --output-dir figures/reference
```

Select figures or supply another result location with:

```bash
python plots/plot_all.py --figures e2e tracea agent
python plots/collect_results.py --raw-root /path/to/raw --output-dir /path/to/summary
python plots/plot_all.py --summary-dir /path/to/summary \
  --kernel-dir /path/to/kernel --output-dir /path/to/figures
```

## 8. Validate the data and plotting workflow

```bash
python scripts/check_replay_inputs.py
python scripts/smoke_test.py
```

The replay input check loads the packaged conversations and answers through the original vLLM and SGLang interfaces. It checks all 24 LongBench/LooGLE schedule files and compares request histories and reference answers using a symbolic tokenizer. Its report is saved to `results/validation/replay-inputs.json`.

The smoke test creates 228 synthetic experiment logs with known timestamps, cache metrics and agent sessions. It runs the family analyzers, checks TTFT/TPOT/hit-rate/job-latency values, checks complete baseline coverage, checks rejection of duplicate results, and renders six PDF/PNG pairs. Synthetic inputs, summaries and figures are stored under `results/smoke/`.
