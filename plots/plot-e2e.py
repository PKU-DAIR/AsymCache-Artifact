import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.patches as mpatches
import argparse

from parse_data import load_data
from plot_hist2 import plot_grouped_bar_chart
from my_colors import *

def parse_args():
    parser = argparse.ArgumentParser(description="Plotting script for E2E evaluation results")
    parser.add_argument('--div', choices=['low', 'high'], default='low', help="dispersion level (default: low)")
    parser.add_argument('--path', type=str, default='../results/summary.csv', help="path to the data file (default: ../results/summary.csv)")
    return parser.parse_args()

args = parse_args()

dataset = load_data(path=args.path, v2=True)
SPLIT_MODE = 'frac' # 'model' or 'frac'

if SPLIT_MODE == 'model':
    figure_model = "70B"
elif SPLIT_MODE == 'frac':
    figure_frac = '5' if args.div == 'low' else '10'

# 1. 设置全局字体为Calibri
mpl.rcParams['font.family'] = 'Calibri'
mpl.rcParams['font.size'] = 12

# 2. 创建图形
fig = plt.figure(figsize=(12, 4), dpi=100)

# 3. 创建外层GridSpec：3行4列
# 宽度比例：第一列:第二列:第三列:第四列
# 这里可以调整每列的相对宽度
ratio = 1.2
outer_width_ratios = [ratio, ratio, 1, 1]  # 第一、二列比第三、四列宽
outer_height_ratios = [1, 1, 1]  # 行高比例

# 外层网格，设置列间距（用于第三列和第四列之间的间距）
outer_gs = GridSpec(
    nrows=3, ncols=4,
    width_ratios=outer_width_ratios,
    height_ratios=outer_height_ratios,
    wspace=0.1,  # 外层列间距（控制第2列与第3列之间，以及第3列与第4列之间的间距）
    hspace=0.1    # 行间距
)

# 4. 创建子图坐标轴
axes = []

# 遍历每一行
for row in range(3):
    row_axes = []
    
    # 第一列：嵌套GridSpec实现2:1分割且无缝
    # 创建内层GridSpec（1行2列，无间距）
    inner_gs1 = GridSpecFromSubplotSpec(
        1, 2, 
        subplot_spec=outer_gs[row, 0],
        width_ratios=[1.8, 1],
        wspace=0  # 无缝连接
    )
    ax_left = fig.add_subplot(inner_gs1[0, 0])
    if row < 2:
        ax_right = fig.add_subplot(inner_gs1[0, 1])
    else:
        ax_right = fig.add_subplot(inner_gs1[0, 1], sharey=ax_left)
    row_axes.extend([ax_left, ax_right])
    
    # 第二列：同样嵌套GridSpec实现2:1分割且无缝
    inner_gs2 = GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=outer_gs[row, 1],
        width_ratios=[1.8, 1],
        wspace=0  # 无缝连接
    )
    ax_left = fig.add_subplot(inner_gs2[0, 0])
    if row < 2:
        ax_right = fig.add_subplot(inner_gs2[0, 1])
    else:
        ax_right = fig.add_subplot(inner_gs2[0, 1], sharey=ax_left)
    row_axes.extend([ax_left, ax_right])
    
    # 第三列：直接创建子图
    ax3 = fig.add_subplot(outer_gs[row, 2])
    row_axes.append(ax3)
    
    # 第四列：直接创建子图
    ax4 = fig.add_subplot(outer_gs[row, 3])
    
    row_axes.append(ax4)
    
    axes.append(row_axes)


y_label_list = ['TTFT (s)', 'TPOT (ms)', 'Hit Rate (%)']
# 5. 设置坐标轴和标签格式
for row in range(3):
    for col_idx in range(6):
        ax = axes[row][col_idx]
        
        # 隐藏顶部和右侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 设置x轴刻度（除最下方行外隐藏标签）
        if row < 2:  # 不是最下方行
            ax.tick_params(axis='x', labelbottom=False)
        # else:  # 最下方行
        #     # 设置x轴标签
        #     if col_idx in [0, 1]:  # 第一列
        #         ax.set_xlabel('X Axis (Column 1)', fontweight='bold')
        #     elif col_idx in [2, 3]:  # 第二列
        #         ax.set_xlabel('X Axis (Column 2)', fontweight='bold')
        #     elif col_idx == 4:  # 第三列
        #         ax.set_xlabel('X Axis (Column 3)', fontweight='bold')
        #     elif col_idx == 5:  # 第四列
        #         ax.set_xlabel('X Axis (Column 4)', fontweight='bold')
        
        # 设置y轴标签（只在每行的第一个子图显示）
        if col_idx == 0:  # 第一列左子图
            ax.set_ylabel(y_label_list[row], fontweight='bold')
        
        # 设置标题（示例）
        # if col_idx == 0:
        #     ax.set_title(f'Row {row+1}, Left Part', fontsize=11, fontweight='bold')
        # elif col_idx == 1:
        #     ax.set_title(f'Row {row+1}, Right Part', fontsize=11, fontweight='bold')
        # elif col_idx == 2:
        #     ax.set_title(f'Row {row+1}, Left Part', fontsize=11, fontweight='bold')
        # elif col_idx == 3:
        #     ax.set_title(f'Row {row+1}, Right Part', fontsize=11, fontweight='bold')
        # elif col_idx == 4:
        #     ax.set_title(f'Row {row+1}, Column 3', fontsize=11, fontweight='bold')
        # elif col_idx == 5:
        #     ax.set_title(f'Row {row+1}, Column 4', fontsize=11, fontweight='bold')

# 6. 特殊处理：合并第一列和第二列的左右子图
def merge_split_axes(left_ax, right_ax, split_line_visible=True):
    """
    将分割的子图视觉上合并为一个图
    """
    # 移除左图的右侧边框
    left_ax.spines['right'].set_visible(False)
    
    # 移除右图的左侧边框，显示右侧边框用于第二个y轴
    right_ax.spines['left'].set_visible(split_line_visible)
    right_ax.spines['right'].set_visible(True)
    right_ax.yaxis.set_ticks_position('right')
    right_ax.yaxis.set_label_position('right')
    
    # 设置右侧y轴标签
    right_ax.set_ylabel('Y Axis Right', fontweight='bold', rotation=-90, labelpad=15)

# 处理第一列的分割图
for row in range(3):
    merge_split_axes(axes[row][0], axes[row][1], row < 2)

# 处理第二列的分割图
for row in range(3):
    merge_split_axes(axes[row][2], axes[row][3], row < 2)
# 7. 隐藏重复的y轴标签
for row in range(3):
    # 第一列右侧子图（已经设置了右侧标签）
    axes[row][1].set_ylabel('')
    
    # 第二列右侧子图
    axes[row][3].set_ylabel('')
    
    # 第三列和第四列（不是最左侧）隐藏y轴标签
    # 只在第一行设置标题，其他行隐藏
    
    axes[row][4].set_ylabel('')
    axes[row][5].set_ylabel('')
    
    # if row != 0:
    #     axes[row][4].set_ylabel('')
    #     axes[row][5].set_ylabel('')
    # else:
    #     axes[row][4].set_ylabel('Y Axis', fontweight='bold')
    #     axes[row][5].set_ylabel('Y Axis', fontweight='bold')

# 8. 调整刻度标签对齐
for row in range(3):
    for col_idx in range(6):
        ax = axes[row][col_idx]
        # 左对齐左侧刻度标签，右对齐右侧刻度标签
        if col_idx in [1, 3]:  # 右侧子图
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
    # x = np.linspace(0, 10, 100)
    
    # if col_idx == 0:  # 第一列左图
    #     y = np.sin(x) * (row + 1)
    #     ax.plot(x, y, 'b-', linewidth=1.5)
    #     ax.set_ylim(-3, 3)
        
    # elif col_idx == 1:  # 第一列右图（共享y轴，但使用不同刻度）
    #     # 这里可以设置不同的数据范围和刻度
    #     y = np.cos(x) * (row + 1) * 10
    #     ax.plot(x, y, 'r-', linewidth=1.5)
    #     ax.set_ylim(-30, 30)
        
    # elif col_idx == 2:  # 第二列左图
    #     y = np.exp(-x/3) * (row + 1)
    #     ax.plot(x, y, 'g-', linewidth=1.5)
        
    # elif col_idx == 3:  # 第二列右图
    #     y = np.log(x + 1) * (row + 1) * 5
    #     ax.plot(x, y, 'm-', linewidth=1.5)
        
    # elif col_idx == 4:  # 第三列
    #     y = np.random.randn(100).cumsum() + row * 10
    #     ax.plot(x, y, 'c-', linewidth=1.5)
        
    # elif col_idx == 5:  # 第四列
    #     y = np.sin(x + row) * np.exp(-x/5)
    #     ax.plot(x, y, 'y-', linewidth=1.5)
    
    if SPLIT_MODE == 'model':
        frac = '5' if col_idx in [0, 1, 4] else '10'
        dataset_name = 'longbench' if col_idx < 4 else 'loogle'
        model = figure_model
    elif SPLIT_MODE == 'frac':
        dataset_name = 'longbench' if col_idx < 4 else 'loogle'
        model = "8B" if col_idx in [0, 1, 4] else "70B"
        frac = figure_frac
    
    # print(data_in, dataset, model, frac)
    
    try:
        data = dataset[dataset_name][model][frac]
    except KeyError as e:
        print(f"[ERROR] 无法找到数据: dataset={dataset_name}, model={model}, frac={frac}")
        return []
    
    # print("Plotting row:", row, "col:", col_idx, "dataset:", dataset_name, "frac:", frac)
    plot_grouped_bar_chart(ax, data, row+1, col_idx+1)
    
    
    if SPLIT_MODE == 'frac':
        if col_idx in [0, 2]:
            ax.set_xlim(0.5, 2.5)
        if col_idx in [1, 3]:
            ax.set_xlim(0.55, 1.5)
        
        # 逐图控制
        if frac == '5':
            if (row, col_idx) == (0, 0):
                ax.set_ylim(0, 15)
            elif (row, col_idx) == (1, 0):
                ax.set_ylim(0, 60)
                
            elif (row, col_idx) == (0, 1):
                ax.set_ylim(0, 650)
            elif (row, col_idx) == (1, 1):
                ax.set_ylim(0, 800)
            elif (row, col_idx) == (2, 1):
                ax.set_ylim(0, 65)
                
            elif (row, col_idx) == (0, 2):
                ax.set_ylim(0, 30)
            elif (row, col_idx) == (1, 2):
                ax.set_ylim(0, 180)
            
            elif (row, col_idx) == (0, 3):
                ax.set_ylim(0, 1200)
            elif (row, col_idx) == (1, 3):
                ax.set_ylim(0, 1400)
            elif (row, col_idx) == (2, 3):
                ax.set_ylim(0, 65)
                
            elif (row, col_idx) == (0, 4):
                ax.set_ylim(0, 8)
            elif (row, col_idx) == (1, 4):
                ax.set_ylim(0, 39)
            elif (row, col_idx) == (2, 4):
                ax.set_ylim(0, 95)
                
            
            elif (row, col_idx) == (0, 5):
                ax.set_ylim(0, 15)
            elif (row, col_idx) == (1, 5):
                ax.set_ylim(0, 180)
            elif (row, col_idx) == (2, 5):
                ax.set_ylim(0, 95)
        elif frac == '10':
            if (row, col_idx) == (0, 0):
                ax.set_ylim(0, 15)
            elif (row, col_idx) == (1, 0):
                ax.set_ylim(0, 60)
                
            elif (row, col_idx) == (0, 1):
                ax.set_ylim(0, 700)
            elif (row, col_idx) == (1, 1):
                ax.set_ylim(0, 800)
            elif (row, col_idx) == (2, 1):
                ax.set_ylim(0, 65)
                
            elif (row, col_idx) == (0, 2):
                ax.set_ylim(0, 30)
            elif (row, col_idx) == (1, 2):
                ax.set_ylim(0, 180)
            
            elif (row, col_idx) == (0, 3):
                ax.set_ylim(0, 1300)
            elif (row, col_idx) == (1, 3):
                ax.set_ylim(0, 1500)
            elif (row, col_idx) == (2, 3):
                ax.set_ylim(0, 65)
                
            elif (row, col_idx) == (0, 4):
                ax.set_ylim(0, 9)
            elif (row, col_idx) == (1, 4):
                ax.set_ylim(0, 65)
            elif (row, col_idx) == (2, 4):
                ax.set_ylim(0, 95)
                
            
            elif (row, col_idx) == (0, 5):
                ax.set_ylim(0, 27)
            elif (row, col_idx) == (1, 5):
                ax.set_ylim(0, 240)
            elif (row, col_idx) == (2, 5):
                ax.set_ylim(0, 95)
        
    if row == 2 and col_idx in [1, 3, 4, 5]:
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
    for col_idx in range(6):
        plot_example_data(axes[row][col_idx], row, col_idx)

patches = []
for approach, label in label_map.items():
    patch = mpatches.Patch(color=get_color_bool(approach, 1, False), label=label)
    patches.append(patch)
fig.legend(handles=patches, loc='upper center', 
           bbox_to_anchor=(0.5, 0.97), ncol=5, 
           fontsize=11,
           framealpha=0.0,)

fig.text(0.23, 0.01, 'Llama3.1-8B + LongBench', 
         ha='center', fontsize=12, fontweight='bold')
fig.text(0.44, 0.01, 'Llama3.1-70B + LongBench', 
         ha='center', fontsize=12, fontweight='bold')
fig.text(0.64, 0.01, 'Llama3.1-8B + LooGLE', 
         ha='center', fontsize=12, fontweight='bold')
fig.text(0.82, 0.01, 'Llama3.1-70B + LooGLE', 
         ha='center', fontsize=12, fontweight='bold')

# 11. 调整布局
plt.tight_layout()

# 12. 显示图形
# plt.show()

# 13. 保存图形（可选）
file_name = f"figure_e2e_f{figure_frac}"
print(f"Saving figure to {file_name}.png and {file_name}.pdf")
fig.savefig(f'{file_name}.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{file_name}.pdf', dpi=300, bbox_inches='tight')