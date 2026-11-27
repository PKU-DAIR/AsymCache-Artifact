# Paper figure generation

Run commands from the artifact root. Summarize all replay outputs, then generate the figures:

```bash
python plots/collect_results.py
python plots/plot_all.py
```

| Input table | Plot command | Output basename |
| --- | --- | --- |
| `results/summary/e2e.csv` | `python plots/plot-e2e.py --div low` | `figure_e2e_f5` |
| `results/summary/e2e.csv` | `python plots/plot-e2e.py --div high` | `figure_e2e_f10` |
| `results/summary/tracea.csv` | `python plots/plot-tracea.py` | `figure_tracea` |
| `results/summary/agent.csv` | `python plots/plot-agent.py` | `figure_agent` |
| `results/summary/sensitivity.csv` and `e2e.csv` | `python plots/plot_sense.py` | `sense` |
| Six CSV files in `results/raw/kernel/` | `python plots/plot-kernel.py` | `ablation-fused2` |

Each plot writes PDF and PNG versions to `figures/`. All entry points accept `--output-dir`. Result-table plots accept `--path`; the sensitivity plot also accepts `--baselines`, and the kernel plot accepts `--data-dir`. Default paths are resolved relative to the artifact checkout.

The result schema includes `dataset`, `model`, `frac`, `approach`, `qps`, `file`, `total_items`, `avg_ttft`, `avg_tpot`, `hit_items`, `avg_hit_rate`, `total_time`, and `throughput`. Agent tables add `avg_job_latency` and `job_latency_p90`. Sensitivity rows identify the parameter setting with `set` and `grad`. Latencies use seconds and hit rates use fractions. The plotters convert TPOT to milliseconds and hit rates to percent.

The collector selects the original analyzer for each replay schema: vLLM end-to-end/sensitivity, vLLM TraceA, SGLang, and BFCL. It uses the experiment path and the end-to-end `run.json` manifest to retain model and workload identity. Completeness checks run before plotting. Duplicate configuration keys raise an error.

To inspect sensitivity panel values as a flat table:

```bash
python plots/build_summary_sense.py
```

The six-panel sensitivity figure selects its LRU references and default AsymCache points from the end-to-end table at LongBench, 8B, QPS 0.02 and the matching dispersion factor. The included paper table supplies the original sensitivity default points explicitly.

To render the included paper measurements:

```bash
python plots/plot_all.py --reference --output-dir figures/reference
```

To validate the complete plotting workflow with synthetic replay data:

```bash
python scripts/smoke_test.py
```
