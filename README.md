# AsymCache Artifact

Please refer to [[PROOF] properties of order-preserving rule.pdf](./[PROOF]%20properties%20of%20order-preserving%20rule.pdf) for the missed proof in Section 4.4.

## 1. Description

Environment dependencies:

```text
CUDA: 12.8
CPU: x86_64
GPU: 8 x H20
PyTorch: 2.7.0
```

This repository is the artifact of AsymCache. Currently, the code has not been fully arranged, but is still able to reproduce all the main results in the paper.

Our modified vllm is a submodule in this repository, which is a public repo at: https://gitee.com/spiritedawaycn/vllm-dev.git

## 2. Installation

Please download this repository, and change your directory using:

```bash
cd multi-segment-attention
```

If the repository has already been cloned without submodules, initialize the submodule manually:

```bash
git submodule update --init --recursive
```

Alternatively, you can manually clone our modifed vllm repo using:
```bash
git clone https://gitee.com/spiritedawaycn/vllm-dev.git
```

After this step, the vllm/ directory should not be empty.

```bash
conda create -n msa python=3.10
conda activate msa
```

Install `pod_attn`:

```bash
cd pod_attn_dev
pip install -e .
```

Install the local vLLM package:

```bash
cd ../vllm
pip install -e . -v -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Build the `c_set` extension:

```bash
cd ./csrc/c_set
g++ -O3 -Wall -shared -std=c++17 -fPIC \
  $(python3 -m pybind11 --includes) \
  c_set.cpp \
  -o c_set$(python3-config --extension-suffix)
```

The SGLang experiments use the customized source in `sglang-dev/`. Set the
paths relative to the repository root:

```bash
export REPO=/path/to/Multi-Segment-Attention
export VLLM_ROOT="${REPO}/vllm"
export SGLANG_ROOT="${REPO}/sglang-dev"
```

Activate the SGLang environment. The original experiments used Python 3.10:

```bash
conda create -n sglang-pod python=3.10 -y
conda activate sglang-pod
python -m pip install -r "${SGLANG_ROOT}/requirements.txt"
```

Install `pod_attn` and build the local vLLM package in this environment by
following the preceding steps. Then install the customized SGLang package:

```bash
cd "${SGLANG_ROOT}"
python -m pip install -e ./python
```

SGLang imports the vLLM FlashAttention extension through
`python/sglang/vllm_flash_attn_org`. Create a symbolic link to the extension
that was built in the local vLLM source tree:

```bash
ln -s "${VLLM_ROOT}/vllm/vllm_flash_attn" \
  "${SGLANG_ROOT}/python/sglang/vllm_flash_attn_org"
```

The resulting link inside the cloned repository should be:

```text
Multi-Segment-Attention/sglang-dev/python/sglang/vllm_flash_attn_org
  -> Multi-Segment-Attention/vllm/vllm/vllm_flash_attn
```

## 3. Run Experiments

### (1) LongBench and LooGLE

Set the workspace path. This should be the directory that contains `vllm-dev`:

```bash
export WORKSPACE=/home
```

Switch to the `simulator` branch and run the simulator experiments:

```bash
cd ${WORKSPACE}/vllm-dev
git switch simulator
cd ${WORKSPACE}/vllm-dev/examples/test/e2e-exp/reproduce
bash simulator_run_all.sh
```

Switch to the `shed` branch and run the scheduler experiments:

```bash
cd ${WORKSPACE}/vllm-dev
git switch shed
cd ${WORKSPACE}/vllm-dev/examples/test/e2e-exp/reproduce
bash shed_run_all.sh
```

The run scripts create per-experiment folders under the shared results directory, such as:

```bash
${WORKSPACE}/results/lru
${WORKSPACE}/results/pensieve
${WORKSPACE}/results/max_score
```

After all experiments finish, run:

```bash
python3 analyse.py
```

The analysis scans the shared `results` directory next to `vllm-dev` and writes the merged CSV summary to:

```bash
${WORKSPACE}/results/summary.csv
```

### (2) BFCL workload

Data files:

```text
multi-segment-attention/BFCL_dataset/2x_bfcl_web_search_prompts_qps1.5.csv
multi-segment-attention/BFCL_dataset/2x_bfcl_web_search_prompts_qps1.75.csv
multi-segment-attention/BFCL_dataset/2x_bfcl_web_search_prompts_qps2.0.csv
```

Before running experiments on a new machine, edit `env.sh` or override its variables from the shell. The important variables are:

```text
PYTHON_BIN            # Python executable in the vLLM environment
MODEL_8B_PATH         # local path to Llama-3.1-8B-Instruct
MODEL_70B_PATH        # local path to Llama-3.1-70B-Instruct
HF_HOME               # Hugging Face cache root
HUGGINGFACE_HUB_CACHE # Hugging Face hub cache
CSET_DIR              # path required by the custom vLLM/pod_attention build
```

Use the shared launcher:

```bash
bash run_bfcl_baseline_server.sh --baseline MWF --model-size 70B --qps 1.75
```

The required command-line arguments are:

```text
--baseline    LRU, MWF, CONTINUUM, or CONTINUUM_MWF
--model-size  8B or 70B
--qps         1.5, 1.75, or 2.0
```

`BFCL_24_EXPERIMENT_COMMANDS.sh` contains the complete list of 24 commands. It is a copy-paste command list, not a script to run end-to-end. Do not run: `bash BFCL_24_EXPERIMENT_COMMANDS.sh`. Run one command at a time, or submit each command as a separate cluster job.

By default, outputs are written under:

```text
./${MODEL_SIZE}_results/2x_${BASELINE}_qps${QPS}/
```

For example:

```text
70B_results/2x_MWF_qps1.75/
|-- 2x_MWF_replay_output_qps1.75.jsonl
|-- schedule_qps1.75.log
|-- server_output_qps1.75.log
```

After the replay JSONL files are produced, summarize all 24 result directories:

```bash
python full_result.py
```

The summary CSV is written to:

```text
final_bfcl_24_summary.csv
```

### (3) traceA workload

For the five vLLM-based baselines, switch to the hotprefix-sim branch:

```bash
cd /path/to/vllm-dev
git switch hotprefix-sim
cd /path/to/vllm-dev/examples/test/e2e-exp/Qwen_trace
```

For each QPS, compute the MWF parameters with `CDF.py` first:

```bash
python CDF.py /path/to/tracea_2k_cache_pressure_relaxed_2100.jsonl --target-qps 0.2 --percentile 75
```

The numeric line in the output is:

```text
<percentile> <minexp-b> <minexp-alpha> <minexp-beta> <minexp-a>
```

Fill the values into `run_traceA_2100.sh`:

```text
DEFAULT_MINEXP_B=...
DEFAULT_MINEXP_ALPHA=...
DEFAULT_MINEXP_BETA=...
```

Then run:

```bash
bash run_traceA_2100.sh --all-policies --model-size 8B --qps 0.2
bash run_traceA_2100.sh --all-policies --model-size 70B --qps 0.2
```

Repeat for `--target-qps 0.25` and `--target-qps 0.3`.

### (4) SGLang baseline

The SGLang baseline only needs to be evaluated for LongBench, LooGLE, and traceA. 

Before running the experiments, update `env.sh` according to the local machine. In particular, please check `SGLANG_ROOT`, `DATA_ROOT`, model paths. Make sure that the workload files are available, and run the scripts inside the SGLang Conda environment.

To run all SGLang baseline experiments:

```bash
cd /path/to/sglang-dev/workload-exp/exp-reproduce
bash run_all.sh
```

The script runs six stages sequentially: If one stage fails, the remaining stages will still be executed, and the script will report the failed stages. All outputs are written to:

```text
RESULTS_ROOT/<dataset>/<8B|70B>/
```

## 4. Hyper-parameter Sensitivity

Switch to the `simulator` branch and run the experiments:

```bash
cd /path/to/vllm-dev
git switch simulator
cd /path/to/vllm-dev/examples/test/e2e-exp/hyper_parameter
```

Each line in `ablation_config.txt` follows this format:

```text
<qps> <set> <grad> <minexp_b> <minexp_alpha> <minexp_beta> <dataset> <frac>
```

For example:

```text
0.02 0.6x 40 -6.456619e+02 6.257895e+04 1.564474e+03 longbench 5
```

Before running, please check the hard-coded paths and machine-specific settings in `run_ab_one.sh`.

Important variables include:

```bash
export CSET_DIR=/path/to/vllm/csrc/c_set
BASE_DIR="/path/to/ablation_results"
PY_VLLM="/path/to/conda/env/bin/python3"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
GPU="0"
PORT=8050
VLLM_PORT=12345
```

Please update these paths according to your local environment. Then, run:

```bash
bash run_ab_sweep.sh
```

For each configuration, `run_ab_one.sh` creates one output folder under:

```text
ablation_results/
```

The folder naming format is:

```text
LongBench_300_overall_qps<qps>_frac<frac>_set<set>_grad<grad>_gamma
```

For example:

```text
LongBench_300_overall_qps0.02_frac5_set0.6x_grad40_gamma
LongBench_300_overall_qps0.02_frac10_set1x1y_grad50_gamma
```

The provided configuration file contains 20 experiment settings, so the final result should contain 20 folders.

Finally, run:

```bash
python result_analyse.py
```

## 5. Plot Figures

Before plotting figure, please first change your directory to:

```bash
cd ${WORKSPACE}/plots
```

### End-to-end Results (Figure 11 & 12)

Please run following command:
```bash
python plot-e2e.py --path ${WORKSPACE}/results/summary.csv --div low  # Figure 11
python plot-e2e.py --path ${WORKSPACE}/results/summary.csv --div high # Figure 12
```

### Qwen Dataset Results (Figure 13)
Currently, the end-to-end result tables are stored separately under the vLLM and SGLang paths. Please manually merge them into a single CSV file by concatenating the rows.

Set the `path` argument of `load_data_tracea` in [plot-tracea.py](plots/plot-tracea.py) to the merged CSV file, then run:

```bash
python plot-tracea.py
```

A one-command plotting workflow will be available soon.

### Hyper-parameter Sensitivity (Figure 15)
To make the plotting script identify which rows belong to which subplot, we provide a preprocessing script:

```bash
python build_summary_sense.py --input /path/to/ablation.csv --output summary-sense.csv
```

After generating summary-sense.csv, please manually check and complete the baseline rows used for comparison. Specifically, for each subplot, the corresponding LRU and MWF baseline results under the same workload configuration should be copied and pasted into the appropriate rows below the corresponding subplot marker.

Please first check whether the data path has been correctly set in `plot_sense.py` (line 13).

Then, run `plot_sense.py` to plot figure (Figure 15 in the paper).

### Integration with Agentic System (Figure 16)

Please first check whether the data path has been correctly set in `plot-agent.py` (line 15).

Then, run `plot-agent.py` to plot figure (Figure 16 in the paper).
