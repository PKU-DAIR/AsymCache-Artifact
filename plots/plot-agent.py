from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

from my_colors import *
from parse_data import load_data_bfcl

SHOW_QPS = ['1.5', '1.75', '2.0']
APPROACHES = ['lru', 'mwf', 'continuum', 'continuum_mwf']
PLOT_TAIL = True
PANEL_ROWS = 3 if PLOT_TAIL else 2

JOB_AVG_YLIM_LEFT = (0, 1199)
JOB_AVG_YLIM_RIGHT = (0, 1949)
JOB_P90_YLIM_LEFT = (0, 2399)
JOB_P90_YLIM_RIGHT = (0, 3999)
HIT_RATE_YLIM_RIGHT = (0, 85)


def get_metric_key(row: int) -> str:
    if PLOT_TAIL:
        if row == 2:
            return 'hit_rate'
        if row == 1:
            return 'job_p90'
    elif row == 1:
        return 'hit_rate'
    return 'job_avg'

def plot_grouped_bar_chart(ax: plt.Axes,
                          data: Union[Dict, List, np.ndarray, Tuple],
                          row: int,
                          col: int,
                          bar_width: float = 0.6,
                          gap_even_to_odd: float = 0.02,
                          gap_odd_to_even: float = 0.3):
    """Draw grouped bars with asymmetric gaps between adjacent bars.

    gap_even_to_odd and gap_odd_to_even are fractions of `bar_width` (total group width).
    The gap between bar i and i+1 uses `gap_even_to_odd` when i is even,
    and `gap_odd_to_even` when i is odd.
    """
    entry_num = len(SHOW_QPS)
    values_matrix = np.zeros((4, entry_num))

    keyname = get_metric_key(row)

    for i, qps in enumerate(SHOW_QPS):
        if qps not in data:
            print(f"Warning: QPS {qps} not found in data for row {row}, col {col}")
            continue
        qps_data = data[qps]
        for j, approach in enumerate(APPROACHES):
            if approach not in qps_data:
                print(f"Warning: Approach {approach} not found in data for QPS {qps} (row {row}, col {col})")
                continue
            values_matrix[j, i] = qps_data[approach][keyname]


    m, n = values_matrix.shape  # m: 类别数, n: x轴点数

    # 计算柱子位置（支持奇偶不同间隔）
    x = np.arange(1, n + 1)  # x轴坐标从1开始

    # initial per-bar raw width (will be adjusted if gaps consume space)
    raw_bar_w = (bar_width * 0.9) / m

    # prepare raw gaps (absolute, as fraction of group width)
    raw_gaps = []
    for gap_index in range(m - 1):
        if gap_index % 2 == 0:
            raw_gaps.append(gap_even_to_odd * bar_width)
        else:
            raw_gaps.append(gap_odd_to_even * bar_width)

    total_raw_gap = sum(raw_gaps)
    total_raw_bar = raw_bar_w * m

    # If gaps + bars exceed bar_width, scale gaps down to fit remaining space
    remaining = bar_width - total_raw_bar
    if remaining <= 0 or total_raw_gap <= 0:
        # no space for gaps -> collapse gaps to zero and recompute bar width to fill group
        gaps = [0.0] * (m - 1)
        bar_w = (bar_width * 0.98) / m
    else:
        gap_scale = remaining / total_raw_gap
        gaps = [g * gap_scale for g in raw_gaps]
        bar_w = raw_bar_w

    bars = []  # 存储柱子对象

    # compute x positions for each category i within a group
    # start at left edge of group: center_x - bar_width/2
    for i in range(m):
        # compute per-category offset from group center
        # For each group x[j], compute leftmost center and then add cumulative offsets
        x_positions = np.zeros(n)
        for j in range(n):
            # leftmost center of group
            left_center = x[j] - bar_width / 2 + bar_w / 2
            # sum widths and gaps for previous bars
            offset = sum(bar_w for _ in range(0))
            # accumulate offsets for bars before i
            cum = 0.0
            for k in range(i):
                cum += bar_w
                if k < len(gaps):
                    cum += gaps[k]
            x_positions[j] = left_center + cum

        for j in range(n):
            bar_container = ax.bar(x_positions[j], values_matrix[i][j],
                                   width=bar_w,
                                   color=agent_color_palette[APPROACHES[i]],
                                   edgecolor='white',
                                   linewidth=0.,
                                   hatch=None)
            bars.append(bar_container)

    # 在奇数列，展示相比前一个偶数列的百分比差值
    if row < PANEL_ROWS - 1:  # 最后一行不展示百分比
        for j in range(n):
            for i in range(1, m, 2):
                ratio = - (values_matrix[i-1][j] / values_matrix[i][j] - 1) * 100
                # compute x position consistent with how bars were placed
                left_center = x[j] - bar_width / 2 + bar_w / 2
                cum = 0.0
                for k in range(i):
                    cum += bar_w
                    if k < len(gaps):
                        cum += gaps[k]
                x_pos = left_center + cum
                y_pos = values_matrix[i][j]
                bar_color = agent_text_color_palette[APPROACHES[i]]
                ax.text(x_pos, y_pos, f' {ratio:+.1f}%',
                        ha='center', va='bottom', fontsize=8,
                        color=bar_color, rotation=90)

    # 设置坐标轴
    # ax.set_xlabel(x_label, fontsize=12)
    # ax.set_ylabel(y_label, fontsize=12)
    # ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    # show QPS labels for each x tick
    ax.set_xticklabels([f'QPS {SHOW_QPS[i]}' for i in range(n)])

    # 添加网格线
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # 添加图例
    # if categories:
    #     ax.legend(title=legend_title, fontsize=10, title_fontsize=11)

    # 自动调整y轴范围
    max_val = values_matrix.max()
    ax.set_ylim(0, max_val * 1.15)

    # 设置坐标轴样式
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return bars


from plot_common import plot_args, save_figure
args = plot_args('agent')
dataset = load_data_bfcl(args.path)

mpl.rcParams["font.family"] = "Calibri"
mpl.rcParams["font.size"] = 11


fig_height = 2.6 if not PLOT_TAIL else 3.2
fig = plt.figure(figsize=(4.6, fig_height), dpi=100)
outer_gs = GridSpec(
    nrows=PANEL_ROWS,
    ncols=2,
    width_ratios=[1, 1],
    height_ratios=[1] * PANEL_ROWS,
    wspace=0.18,
    hspace=0.12,
)

axes = [[None for _ in range(2)] for _ in range(PANEL_ROWS)]
for row in range(PANEL_ROWS):
    for col in range(2):
        if row == PANEL_ROWS-1 and col == 1:
            # share y-axis with the left subplot on the same row
            axes[row][col] = fig.add_subplot(outer_gs[row, col], sharey=axes[row][0])
        else:
            axes[row][col] = fig.add_subplot(outer_gs[row, col])


def style_panel(ax, hide_xtick: bool) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if hide_xtick:
        ax.tick_params(axis="x", labelbottom=False)

    ax.tick_params(axis="x", direction="out", pad=1)
    ax.tick_params(axis="y", direction="inout", pad=-1, left=False, right=False)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    plt.setp(ax.get_xticklabels(), fontsize=9)


for idx in range(PANEL_ROWS * 2):
    grid_row = idx // 2
    grid_col = idx % 2

    ax = axes[grid_row][grid_col]
    data = dataset['8b'] if grid_col < 1 else dataset['70b']
    plot_grouped_bar_chart(ax, data, grid_row, grid_col)

    ax.set_xlim(0.5, len(SHOW_QPS) + 0.5)
    metric_key = get_metric_key(grid_row)

    if grid_col == 0:
        if metric_key == "hit_rate":
            y_label = "Hit Rate (%)"
        else:
            latency_prefix = "P90" if metric_key == "job_p90" else "AVG"
            y_label = rf"{latency_prefix} Job Time (s)"
        ax.text(
            -0.16,
            0.5,
            y_label,
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontweight="bold",
            clip_on=False,
            fontsize=8.5,
        )

    if metric_key == "job_avg":
        if grid_col == 0:
            ax.set_yticks([0, 500])
            ax.set_ylim(*JOB_AVG_YLIM_LEFT)
        else:
            ax.set_ylim(*JOB_AVG_YLIM_RIGHT)
    elif metric_key == "job_p90":
        if grid_col == 0:
            # ax.set_yticks([0, 500])
            ax.set_ylim(*JOB_P90_YLIM_LEFT)
        else:
            ax.set_ylim(*JOB_P90_YLIM_RIGHT)
    elif metric_key == "hit_rate" and grid_col == 1:
        ax.set_ylim(*HIT_RATE_YLIM_RIGHT)

    style_panel(ax, grid_row < PANEL_ROWS - 1)


patches = []
for approach, label in agent_label_map.items():
    patch = mpatches.Patch(color=agent_color_palette[approach], label=label)
    patches.append(patch)

fig.legend(
    handles=patches,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0 if PLOT_TAIL else 1.04),
    ncol=2,
    fontsize=10,
    framealpha=0.0,
    labelspacing=0.2,
    handletextpad=0.6,
    columnspacing=1.5,
)

fig.text(
    0.29,
    0.05 if PLOT_TAIL else 0.01,
    "Llama 3.1 - 8B",
    ha="center",
    fontsize=11,
    fontweight="bold",
)
fig.text(
    0.78,
    0.05 if PLOT_TAIL else 0.01,
    "Llama 3.1 - 70B",
    ha="center",
    fontsize=11,
    fontweight="bold",
)

fig.subplots_adjust(left=0.09, right=0.995, bottom=0.13, top=0.87, wspace=0.18, hspace=0.12)

file_name = "figure_agent"
print(f"Saving figure to {file_name}.png and {file_name}.pdf")
save_figure(fig, args.output_dir, file_name, pad_inches=0.02)
