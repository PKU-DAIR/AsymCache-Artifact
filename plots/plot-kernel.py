import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

import numpy as np
from typing import Tuple, Optional

def calculate_curve_difference_mean(
    x1: np.ndarray, y1: np.ndarray,
    x2: np.ndarray, y2: np.ndarray,
    N: int = 10000
) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    计算两条离散曲线在定义域交集内的平均差异

    参数:
    x1, y1: 第一条曲线的x和y坐标数组
    x2, y2: 第二条曲线的x和y坐标数组
    N: 分割点数，默认为10000

    返回:
    mean_diff: 差异的平均值（如果交集存在）
    x_common: 交集内的x坐标（如果交集存在）
    y_diff: 交集内的y坐标差异（如果交集存在）
    """
    # 1. 计算定义域交集S
    # 获取每条曲线的定义域[min_x, max_x]
    min_x1, max_x1 = np.min(x1), np.max(x1)
    min_x2, max_x2 = np.min(x2), np.max(x2)

    # 计算交集
    S_min = max(min_x1, min_x2)
    S_max = min(max_x1, max_x2)

    # 检查交集是否存在
    if S_min >= S_max:
        print("警告: 两条曲线的定义域没有交集")
        return None, None, None

    print(f"定义域交集: [{S_min}, {S_max}]")
    print(f"交集长度: {S_max - S_min}")

    # 2. 将交集S均分为N个点
    x_common = np.linspace(S_min, S_max, N)

    # 3. 使用线性插值估算两个曲线在x_common处的y坐标
    # 确保x数组是升序排列的（线性插值需要）
    # 首先对两条曲线的数据进行排序
    sorted_idx1 = np.argsort(x1)
    x1_sorted = x1[sorted_idx1]
    y1_sorted = y1[sorted_idx1]

    sorted_idx2 = np.argsort(x2)
    x2_sorted = x2[sorted_idx2]
    y2_sorted = y2[sorted_idx2]

    # 使用numpy的interp函数进行线性插值
    y1_interp = np.interp(x_common, x1_sorted, y1_sorted)
    y2_interp = np.interp(x_common, x2_sorted, y2_sorted)

    # 4. 计算差异
    y_diff = y1_interp - y2_interp

    # 5. 计算差异的均值
    mean_diff = np.mean(y_diff)

    return mean_diff, x_common, y_diff

map_str = {
    'origin': '',
    '2-kernel': '_2kernel',
    'fused:': '_2seg',
}

label_map = {
    'origin': 'From Prefix',
    '2-kernel': 'From Suffix (2 kernel)',
    'fused:': 'From Suffix (MSA)',
}
color_map = {
    '2-kernel': "#be6284",
    'fused:': "#77c474",
    'origin': "#6094c5",
}


from plot_common import ROOT, RESULTS, save_figure
import argparse
from pathlib import Path
parser = argparse.ArgumentParser(description='Plot the MSA kernel comparison')
parser.add_argument('--data-dir', type=Path, default=RESULTS / 'raw/kernel')
parser.add_argument('--output-dir', type=Path, default=ROOT / 'figures')
args = parser.parse_args()
DATA_DIR = args.data_dir

tp = 1

def plot_cache_rate(ax, filename_base, title):
    line_dict = {}
    lines = []  # 用于存储线条对象
    labels = []  # 用于存储标签

    for label, fn in map_str.items():
        try:
            df = pd.read_csv(filename_base%fn)
        except FileNotFoundError:
            print(f'File not found: {filename_base%fn}')
            continue
        # sort by cached_length
        df = df.sort_values(by='cached_length')
        x, y = df['cached_length'], df['AVG']
        line, = ax.plot(x, y, label=label, linewidth=1, color=color_map[label])  # 获取线条对象

        line_dict[label] = (x.values, y.values)  # 存储线条对象到字典中

        lines.append(line)  # 存储线条对象
        labels.append(label_map[label])  # 存储标签

    ax.set_xlabel('Cached Length')
    ax.text(0.58, 0.85, title, ha='center', va='bottom', transform=ax.transAxes,
            fontsize=12, weight='bold')
    # ax.set_title(title)

    diff1, _, _ = calculate_curve_difference_mean(*line_dict['origin'], *line_dict['fused:'])
    diff2, _, _ = calculate_curve_difference_mean(*line_dict['2-kernel'], *line_dict['fused:'])

    # matplotlib 默认颜色
    # 0 -> #1f77b4
    # 1 -> #ff7f0e
    # 2 -> #2ca02c
    # 3 -> #d62728
    if diff1 < 0:
        ax.text(0.25, 0.3, f"{diff1:+.2f}", ha='center', va='bottom', transform=ax.transAxes,
                fontsize=14, color=color_map['origin'], weight='bold')
        ax.text(0.25, 0.4, f"{diff2:+.2f}", ha='center', va='bottom', transform=ax.transAxes,
                fontsize=14, color=color_map['2-kernel'], weight='bold')
    else:
        ax.text(0.25, 0.4, f"{diff1:+.2f}", ha='center', va='bottom', transform=ax.transAxes,
                fontsize=14, color=color_map['origin'], weight='bold')
        ax.text(0.25, 0.3, f"{diff2:+.2f}", ha='center', va='bottom', transform=ax.transAxes,
                fontsize=14, color=color_map['2-kernel'], weight='bold')

    print(f"{title} - diff1 {diff1}, diff2 {diff2}")


    return lines, labels  # 返回线条和标签

mpl.rcParams['font.family'] = 'Calibri'
fig, axs = plt.subplots(1, 2, figsize=(7, 2.3), dpi=100)

# 调整子图之间的间距，减小水平间距
plt.subplots_adjust(wspace=0.0, top=0.85)  # 减小wspace，增加top为图例留空间

all_lines = []  # 存储所有线条
all_labels = []  # 存储所有标签

for i, l in enumerate(['1k', '10k']):
    filename_base = os.path.join(DATA_DIR, f'profile_cache%s_8B_{l}-tp{tp}.csv')
    lines, labels = plot_cache_rate(axs[i], filename_base, f'Prefill Length = {l}')

    # 如果是第一个子图，收集线条和标签
    if i == 0:
        all_lines = lines
        all_labels = labels
        axs[0].set_ylabel('Average Prefill Time (ms)', fontsize=12)

handles = [
    mpatches.Patch(color=color_map[key], label=label_map[key])
    for key in map_str.keys()
]

# 创建一个共同的图例，放在整个图形的正上方
fig.legend(handles=handles,
           loc='upper center',
           ncol=len(map_str),  # 水平排列图例项
           bbox_to_anchor=(0.5, 0.96),  # 将图例定位在图形上方中心
           fancybox=False,
           framealpha=0.0,
           labelspacing=0.2,        # 更紧密的垂直间距
            handletextpad=0.2,       # 更小的水平间距
            handlelength=1.2,        # 更短的图例句柄
            borderpad=0.1,           # 更少的边框内填充
            prop={'size': 12},
           shadow=False)

save_figure(fig, args.output_dir, 'ablation-fused2')  # 使用bbox_inches='tight'确保图例被完整保存
