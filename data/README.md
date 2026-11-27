# Experiment data

All experiment inputs live under `data/workloads/`. Paper result measurements live under `data/reference/paper/`. Generated replay logs, summaries and figures use the `results/` and `figures/` directories described in the root README.

## Conversation inputs

The prepared conversations and generated reference answers are included in the repository. Validate them with:

```bash
python scripts/prepare_data.py
```

The packaged inputs have this layout:

```text
data/workloads/
  Longbench/*.jsonl
  LooGLE_merged/*.jsonl
  data_preprocess/
    longbench_deepseek_results.jsonl
    loogle_deepseek_results.jsonl
```

Each LongBench record contains `input`, `instructions`, and `outputs`. Each LooGLE record contains `context`, `instructions`, and `outputs`. `instructions` and `outputs` are arrays indexed by turn. JSONL filenames and line order match the arrival schedule's `Dataset Name` and zero-based `Data Index`.

Each reference output record contains `file`, `line`, `turn`, and `generate_text`. `line` and `turn` are zero-based. The validator checks reference answers for every turn used to reconstruct a conversation history.

| Dataset | Conversation files | Conversations | Generated reference answers | Schedule files |
| --- | --- | --- | --- | --- |
| LongBench | 6 | 380 | 1,140 | 12 |
| LooGLE | 1 | 140 | 1,101 | 12 |

LongBench files are `2wikimqa.jsonl`, `hotpotqa.jsonl`, `multifieldqa_en.jsonl`, `musique.jsonl`, `narrativeqa.jsonl`, and `qasper.jsonl`. LooGLE uses `longdep_qa.jsonl`. All schedules use the packaged filenames and zero-based record ordering. Each dataset's 12 schedules contain 3,600 request rows in total, including both model sizes and dispersion factors.

The same reference answer is selected by `(file, line, turn)` in the vLLM end-to-end, vLLM sensitivity and SGLang interfaces. The offline interface check compares their constructed requests on the packaged data:

```bash
python scripts/check_replay_inputs.py
```

To import another prepared bundle with the same layout:

```bash
python scripts/prepare_data.py --source /path/to/prepared/datasets --write-manifest
```

The importer copies the conversation JSONL files and the two named reference files. It also imports matching replay schedules, TraceA files and BFCL inputs when present. It retains record ordering and imports the reference filenames consumed by the interfaces.

## Arrival schedules

`longbench_data/` and `loogle_data/` contain CSV files named:

```text
outputs_gamma_qps<5|10>_CV0.25_<8B|70B>_<0.01|0.02|0.04>_subset.csv
```

The `qps5` and `qps10` filename components encode dispersion factors. The final numeric component encodes the request arrival rate. CSV columns are `Timestamp`, `Dataset Name`, `Data Index`, and `Turn`. Timestamps are replay offsets in seconds. The LooGLE 8B and 70B schedules contain the same conversation ordering and arrival offsets.

## TraceA: Qwen-Bailian

`tracea/tracea_2k_cache_pressure_relaxed_2100.jsonl` contains the 2,100-request real workload. Its adjacent `mapping_config.json` supplies the token mapping used by both serving engines. The TraceA launcher derives a separate MWF parameter file for each requested QPS.

## BFCL agent workload

`bfcl/2x_bfcl_web_search_prompts_qps<QPS>.csv` contains the prepared BFCL v4 Web Search workload at QPS 1.5, 1.75, and 2.0. All four agent baselines replay the same corresponding CSV.

## Paper measurements

The four CSV tables in `reference/paper/` use named result columns and fractional hit rates. `kernel/` contains the six original cached-length profiling tables. `reference/paper/provenance.json` records source table names and SHA-256 hashes of the paper project's source files. `workloads.provenance.json` records the source paths and record counts of the prepared conversation bundle. `workloads.sha256.json` records hashes of all 38 packaged workload files.

```bash
python plots/plot_all.py --reference --output-dir figures/reference
```
