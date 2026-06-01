import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import numpy as np
from parse_data import load_data_sense

import matplotlib.patches as mpatches
import matplotlib.lines as mlines

colors = "#0e970a","#2976be","#cb205f"

# 创建6个子图的数据
dataset = load_data_sense(path='../results/summary-sense.csv')

def plot_dual_axis_with_labels(ax, data1, data2, labels, hline_y, 
                               title="Dual Axis Plot", left_figure=True):
    """
    在单个子图上绘制双y轴折线图并标注文本
    
    参数:
    ax: matplotlib轴对象
    data1: 左侧y轴数据 (长度N)
    data2: 右侧y轴数据 (长度N)
    labels: 标注文本列表 (长度N)
    title: 子图标题
    """
    N = len(data1)
    x = np.arange(1, N + 1)  # x坐标为1到N
    
    # 绘制第一条线（左侧y轴）
    line1, = ax.plot(x, data1, linestyle='-', marker='o', label='Line 1', color=colors[1])
    # ax.set_xlabel('X')
    ax.set_ylabel('TTFT', color=colors[1])
    
    # 设置左侧y轴的刻度标签在右侧（内侧）
    ax.tick_params(axis='y', labelcolor=colors[1], direction='in')
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()  # 刻度在左侧
    # 将左侧y轴的刻度标签移动到右侧
    for label in ax.get_yticklabels():
        label.set_horizontalalignment('left')
    
    # ax.set_xticks(x)
    ax.set_xticklabels([])
    
    # 创建第二个y轴
    ax2 = ax.twinx()
    line2, = ax2.plot(x, data2, linestyle='-', marker='s', label='Line 2', color=colors[2])
    ax2.set_ylabel('TPOT', color=colors[2])
    
    
    # 添加文本标注到第一条线的数据点
    for i, (xi, yi, label) in enumerate(zip(x, data2, labels)):
        # 根据位置调整文本偏移，避免重叠，字体为斜体
        offset = 0.12 * max(data2)
        ax2.text(xi, yi - offset, f"{label*100:.1f}%", 
                ha='center', va='top', fontsize=12, color=colors[0], style='italic'
                ) 
    
    # 设置右侧y轴的刻度标签在左侧（内侧）
    ax2.tick_params(axis='y', labelcolor=colors[2], direction='in')
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()  # 刻度在右侧
    # 将右侧y轴的刻度标签移动到左侧
    for label in ax2.get_yticklabels():
        label.set_horizontalalignment('right')
    
    ax.yaxis.set_tick_params(pad=-3)  # 减小左侧y轴刻度标签的间距
    ax2.yaxis.set_tick_params(pad=-3)  # 减小右侧y轴刻度标签的间距
    
    if left_figure:
        ax2.yaxis.set_label_coords(0.93, 0.5)
    else:
        ax.yaxis.set_label_coords(0.07, 0.5)
    
    # 设置两个y轴的刻度线都在坐标轴内侧
    ax.tick_params(axis='both', direction='in')
    ax2.tick_params(axis='both', direction='in')
    
    ax.axhline(y=hline_y['ttft'], color=colors[1], linestyle='--', linewidth=1, alpha=0.7)
    ax2.axhline(y=hline_y['tpot'], color=colors[2], linestyle='--', linewidth=1, alpha=0.7)
    
    # 添加图例
    lines = [line1, line2]
    # ax.legend(lines, [l.get_label() for l in lines], loc='upper left')
    
    # ax.set_title(title)
    # ax.grid(True, alpha=0.3)
    return ax, ax2


mpl.rcParams['font.family'] = 'Calibri'

# 使用 GridSpec 创建自定义间距的布局
fig = plt.figure(figsize=(6.5, 6))
gs = gridspec.GridSpec(5, 2, figure=fig, height_ratios=[1, 0.2, 1, 0.05, 1],
                       hspace=0, wspace=0.0)

# 创建子图列表，第一行和第二行间隔大(0.3)，第二行和第三行间隔小(0.1)
axes = []
axes.append(fig.add_subplot(gs[0, 0]))
axes.append(fig.add_subplot(gs[0, 1]))
axes.append(fig.add_subplot(gs[2, 0]))
axes.append(fig.add_subplot(gs[2, 1]))
axes.append(fig.add_subplot(gs[4, 0]))
axes.append(fig.add_subplot(gs[4, 1]))

# fig.suptitle('Dual Axis Line Plots with Annotations', fontsize=16, y=0.95)

# 遍历所有子图并绘制
for idx, ax in enumerate(axes):
    _, ax2 = plot_dual_axis_with_labels(
        ax, 
        *dataset[idx + 1],  # data1, data2
        # labels=[f'P{i+1}' for i in range(len(dataset[idx + 1][0]))],
        hline_y=dataset[0][idx % 2],  # 使用LRU数据作为参考线
        title=f'Subplot {idx+1}',
        left_figure=(idx % 2 == 0)
    )
    
    if idx in [0, 2, 4]:
        ax.set_ylim(0, 9)
        ax.set_yticks([0, 8.5])
        ax.set_yticklabels(['0', '9'])
        ax2.set_ylim(0, 0.074)
        ax2.set_yticks([0, 0.07])
        ax2.set_yticklabels(['0', '0.07'])
    else:
        ax.set_ylim(0, 11.7)
        ax.set_yticks([0, 11])
        ax.set_yticklabels(['0', '11'])
        ax2.set_ylim(0, 0.085)
        ax2.set_yticks([0, 0.08])
        ax2.set_yticklabels(['0', '0.08'])
    if idx < 2:
        ax.set_xlim(0.5, 5.5)
        ax2.set_xlim(0.5, 5.5)
    else:
        ax.set_xlim(0.5, 4.5)
        ax2.set_xlim(0.5, 4.5)
    
    row = idx // 2
    col = idx % 2
    if row == 0:
        ax.set_title('Low-Dispersion' if col == 0 else 'High-Dispersion', fontsize=14)
        ax.text(0.5, 0.95, 'Choice of Lifespan', 
                ha='center', va='top', transform=ax.transAxes,
                fontsize=12, weight='bold')
        x_label = ['-0.4x', '-0.2x', '0', '+0.2x', '+0.4x']
    elif row == 1:
        ax.text(0.5, 0.95, 'Choice of Reuse Prob.', 
                ha='center', va='top', transform=ax.transAxes,
                fontsize=12, weight='bold')
        x_label = ['0.8', '0.9', '0.95', '0.99']
    elif row == 2:
        ax.text(0.5, 0.95, 'Choice of Changing Ratio', 
                ha='center', va='top', transform=ax.transAxes,
                fontsize=12, weight='bold')
        x_label = ['10', '25', '40', '50']
    
    for i, label in enumerate(x_label):
        ax.text(i+1, 0.2, label,
                ha='center', va='bottom', fontsize=11)


handles = [
    mlines.Line2D([], [], color=colors[1], marker='o', linestyle='-', label='TTFT'),
    mlines.Line2D([], [], color=colors[2], marker='s', linestyle='-', label='TPOT'),
    mlines.Line2D([], [], color=colors[1], linestyle='--', label='LRU TTFT'),
    mlines.Line2D([], [], color=colors[2], linestyle='--', label='LRU TPOT'),
]

plt.legend(handles=handles, loc='center', bbox_to_anchor=(0, 2.15),
           ncol=4, fontsize=11,
           facecolor='lightgray', edgecolor='black', framealpha=0.5)

plt.savefig('sense.pdf', dpi=300, bbox_inches='tight')
plt.savefig('sense.png', dpi=300, bbox_inches='tight')