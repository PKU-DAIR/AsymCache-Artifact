import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.patches as mpatches

from parse_data import load_data_tracea
from plot_hist2 import plot_grouped_bar_chart_tracea
from my_colors import *

from plot_common import plot_args, save_figure
args = plot_args('tracea')
dataset = load_data_tracea(path=args.path)
# print(dataset)

# 1. 设置全局字体为Calibri
mpl.rcParams['font.family'] = 'Calibri'
mpl.rcParams['font.size'] = 12

# 2. 创建图形
fig = plt.figure(figsize=(5, 4), dpi=100)

# 3. 创建外层GridSpec：3行2列
# 宽度比例：第一列:第二列
# 这里可以调整每列的相对宽度
outer_width_ratios = [1, 1]
outer_height_ratios = [1, 1, 1]

# 外层网格，设置列间距
outer_gs = GridSpec(
    nrows=3, ncols=2,
    width_ratios=outer_width_ratios,
    height_ratios=outer_height_ratios,
    wspace=0.1,  # 列间距
    hspace=0.1    # 行间距
)

# 4. 创建子图坐标轴
axes = []

# 遍历每一行
for row in range(3):
    row_axes = []

    # 第一列：直接创建子图
    ax1 = fig.add_subplot(outer_gs[row, 0])
    row_axes.append(ax1)

    # 第二列：直接创建子图
    ax2 = fig.add_subplot(outer_gs[row, 1])
    row_axes.append(ax2)

    axes.append(row_axes)


y_label_list = ['TTFT (s)', 'TPOT (ms)', 'Hit Rate (%)']
# 5. 设置坐标轴和标签格式
for row in range(3):
    for col_idx in range(2):
        ax = axes[row][col_idx]

        # 隐藏顶部和右侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 设置x轴刻度（除最下方行外隐藏标签）
        if row < 2:  # 不是最下方行
            ax.tick_params(axis='x', labelbottom=False)

        # 设置y轴标签（只在每行的第一个子图显示）
        if col_idx == 0:  # 第一列左子图
            ax.set_ylabel(y_label_list[row], fontweight='bold')

# 7. 隐藏重复的y轴标签
for row in range(3):
    # 第一列右侧子图（已经设置了右侧标签）
    axes[row][1].set_ylabel('')

# 8. 调整刻度标签对齐
for row in range(3):
    for col_idx in range(2):
        ax = axes[row][col_idx]
        # 左对齐左侧刻度标签，右对齐右侧刻度标签
        if col_idx == 1:  # 右侧子图
            if row < 2:
                # 不要刻度线，但显示标签
                ax.tick_params(axis='y', direction='inout', pad=2, left=False, right=False)
                plt.setp(ax.get_yticklabels(), ha='right')
            else:
                # 不显示刻度
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
        else:  # 左侧子图
            ax.tick_params(axis='y', direction='inout', pad=-1, left=False, right=False)
            plt.setp(ax.get_yticklabels(), ha='right')

        # 调整刻度标签字体大小
        plt.setp(ax.get_yticklabels(), fontsize=10)
        plt.setp(ax.get_xticklabels(), fontsize=10)

# 9. 示例数据绘制函数（请根据您的需求修改）

cache_leg_handles = [
    mpatches.Patch(facecolor="#b2b2b2", linewidth=0.2, edgecolor='black', label='Block'),
    mpatches.Patch(facecolor="#d4d4d4", linewidth=0, edgecolor='black', label='Request')
]

def plot_example_data(ax, row, col_idx):
    """示例绘图函数，您需要根据实际情况修改"""
    dataset_name = 'tracea'
    model = "8B" if col_idx==0 else "70B"

    # print(data_in, dataset, model, frac)

    try:
        data = dataset[dataset_name][model]
    except KeyError as e:
        print(f"[ERROR] 无法找到数据: dataset={dataset_name}, model={model}")
        return []

    plot_grouped_bar_chart_tracea(ax, data, row+1, col_idx+1)

    if row == 0:
        if col_idx == 0:
            ax.set_ylim(0, 1.9)
        elif col_idx == 1:
            ax.set_ylim(0, 7.4)

    if row == 1:
        if col_idx == 0:
            ax.set_ylim(0, 59)
            ax.set_yticks([20, 40])
        elif col_idx == 1:
            ax.set_ylim(0, 199)

    if row == 2:
        ax.set_ylim(0, 109)
        ax.set_yticks([50])

    if row == 2:
        ax.legend(handles=cache_leg_handles,
                  loc='upper right',
                  prop={'size': 9, 'style': 'italic'},  # 字体更小
                    labelspacing=0.1,        # 更紧密的垂直间距
                    handletextpad=0.1,       # 更小的水平间距
                    handlelength=1.2,        # 更短的图例句柄
                    borderpad=0.0,           # 更少的边框内填充
                    framealpha=0.0
                  )

# 10. 调用绘图函数
for row in range(3):
    for col_idx in range(2):
        plot_example_data(axes[row][col_idx], row, col_idx)

patches = []
for approach, label in label_map.items():
    if approach == 'minexp':
        continue  # 跳过 minexp
    patch = mpatches.Patch(color=get_color_bool(approach, 1, False), label=label)
    patches.append(patch)

patches_transpose = [None] * len(patches)
for i, patch in enumerate(patches):
    target_index = (i % 3) * 2 + (i // 3)
    patches_transpose[target_index] = patch

fig.legend(handles=patches_transpose, loc='upper center',
           bbox_to_anchor=(0.5, 0.96), ncol=3,
           fontsize=10,
           framealpha=0.0,
           labelspacing=0.1,
           handletextpad=0.2,
           columnspacing=0.95,
           borderpad=0.0)

fig.text(0.3, 0.04, 'Llama3.1-8B',
         ha='center', fontsize=12, fontweight='bold')
fig.text(0.7, 0.04, 'Llama3.1-70B',
         ha='center', fontsize=12, fontweight='bold')

# 11. 调整布局
plt.tight_layout()

# 12. 显示图形
# plt.show()

# 13. 保存图形（可选）
file_name = "figure_tracea"
print(f"Saving figure to {file_name}.png and {file_name}.pdf")
save_figure(fig, args.output_dir, 'figure_tracea')
