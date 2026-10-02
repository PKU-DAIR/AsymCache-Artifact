import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Dict, List, Tuple, Any

from my_colors import get_color, get_color_bool

apporaches = ['lru', 'sglang', 'bs', 'pensi', 'hotprefix', 'minexp', 'shed']
apporach_map = {
        'lru': 0,
        'sglang': 1,
        'bs': 2,
        'pensi': 3,
        'hotprefix': 4,
        'minexp': 5,
        'shed': 6,
}
qps_list = {
    2 : [0.01, 0.02],
    3 : [0.01, 0.02, 0.04],
    1 : [0.04],
}
def plot_grouped_bar_chart(ax: plt.Axes, 
                          data: Union[Dict, List, np.ndarray, Tuple],
                          row: int,
                          col: int,
                          x_label: str = "X轴",
                          y_label: str = "Y轴",
                          title: str = "分组柱状图",
                          bar_width: float = 0.7,
                          colors: List[str] = None,
                          legend_title: str = "类别",
                          show_values: bool = False,
                          value_format: str = "{:.1f}"):
    
    entry_num = 3
    if col < 5:
        entry_num = (col % 2) + 1
    values_matrix = np.zeros((
        len(apporaches) - (1 if col in [1, 3, 5, 6] else 0), 
        entry_num))
    values_matrix2 = np.zeros_like(values_matrix)
    
    for (apporach, vals) in data.items():
        for i, qps in enumerate(qps_list[entry_num]):
            try:
                entry = vals[qps]
            except KeyError:
                print(f"[WARN] 缺少数据: approach={apporach}, qps={qps}. vals={vals.keys()}")
                continue
            
            if row == 1:  # TTFT
                values_matrix[apporach_map[apporach]][i] = entry['ttft']
            elif row == 2:  # TPOT
                values_matrix[apporach_map[apporach]][i] = entry['tpot'] * 1000  # 转换为毫秒
            elif row == 3:  # Cache Hit Rate
                values_matrix[apporach_map[apporach]][i] = entry['hit_rate'] * 100  # 转换为百分比
                values_matrix2[apporach_map[apporach]][i] = entry['hit_items'] / 300 * 100
            
            
    m, n = values_matrix.shape  # m: 类别数, n: x轴点数
    
    # 设置颜色
    # if colors is None:
    #     # 使用tab10颜色映射，确保有足够的颜色
    #     colors = plt.cm.tab10(np.linspace(0, 1, min(m, 10)))
    #     if m > 10:
    #         # 如果类别超过10个，使用hsv颜色映射
    #         colors = plt.cm.hsv(np.linspace(0, 0.9, m))
    
    # 计算柱子位置
    x = np.arange(1, n + 1)  # x轴坐标从1开始
    bar_spacing = bar_width / m  # 每个柱子的宽度
    offset = bar_width / 2 - bar_spacing / 2  # 初始偏移量
    
    # 确定要高亮的类别
    if row < 3:
        # 找最小值所在的类别
        flat_idx = np.argmin(values_matrix)
        highlight_idx = flat_idx // n
    else:
        # 找最大值所在的类别
        flat_idx = np.argmax(values_matrix)
        highlight_idx = flat_idx // n
    
    highlight_map = np.zeros_like(values_matrix, dtype=bool)
    # 每一列最大值所在位置应该被高亮
    for j in range(n):
        col_vals = values_matrix[:, j]
        if row < 3:
            max_idx = np.argmin(col_vals)
        else:
            max_idx = np.argmax(col_vals)
        highlight_map[max_idx, j] = True
    
    bars = []  # 存储柱子对象
    
    # 绘制每个类别的柱子
    for i in range(m):
        # 计算当前类别柱子的x位置
        x_positions = x - offset + i * bar_spacing
        
        bars_this = []
        
        for j in range(n):
            # 根据是否为高亮类别设置hatch
            # hatch = '////' if highlight_map[i, j] else None
            hatch = '////' if highlight_map[i, j] else None
        
        
        # 绘制柱子
            if row == 3:
                bar_container = ax.bar(x_positions[j], values_matrix2[i][j], 
                                width=bar_spacing * 0.9,  # 稍微缩小柱子宽度，留出间距
                                color=get_color_bool(apporaches[i], 3, m < len(apporaches)),
                                edgecolor='white',
                                # hatchcolor='white',
                                linewidth=0.0)
                ax.bar(x_positions[j], values_matrix[i][j], 
                                    width=bar_spacing * 0.9,  # 稍微缩小柱子宽度，留出间距
                                    color=get_color_bool(apporaches[i], 1, m < len(apporaches)),
                                    edgecolor='gray',
                                    # hatchcolor='white',
                                    linewidth=0.3,
                                    hatch=hatch)
            else:
                bar_container = ax.bar(x_positions[j], values_matrix[i][j], 
                                    width=bar_spacing * 0.9,  # 稍微缩小柱子宽度，留出间距
                                    color=get_color_bool(apporaches[i], 1, m < len(apporaches)),
                                    edgecolor='white',
                                    # hatchcolor='white',
                                    linewidth=0.,
                                    hatch=hatch)
            # bar_container.set_hatch_color('white')
            bars.append(bar_container)
            bars_this.append(bar_container)
        
        # 显示数值
        if show_values:
            for rect, val in zip(bar_container, values_matrix[i]):
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height + max(values_matrix.max())*0.01,
                       value_format.format(val),
                       ha='center', va='bottom', fontsize=8)
        
    # 对于row<3，在非最小值类别的柱子上方添加倍数标签
        # if row < 3 and i != highlight_idx:
        #     bar_color = get_color_bool(apporaches[i], 2, m < 5)
        #     for j, rect in enumerate(bar_container):
        #         height = rect.get_height()
        #         min_height = values_matrix[highlight_idx][j]
        #         if min_height > 0:  # 避免除以零
        #             ratio = height / min_height
        #             x_pos = rect.get_x() + rect.get_width() / 2.
        #             y_pos = height # + max(values_matrix.max()) * 0.02
        #             ax.text(x_pos, y_pos, f' {ratio:.2f}x',
        #                     ha='center', va='bottom', fontsize=8,
        #                     color=bar_color, rotation=90)
        
        
        if row == 3:
            # 在命中率柱子上方添加命中项百分比标签
            bar_color = get_color_bool(apporaches[i], 2, m < 5)
            # print(f"len of bar_container: {len(bar_container)}, values_matrix2[i]: {values_matrix2[i]}")
            for j, rects in enumerate(bars_this):
                rect = rects.patches[0]  # 获取当前柱子对象
                hit_items_pct = values_matrix2[i][j]
                if hit_items_pct < 2.5:
                    height = rect.get_height()
                    x_pos = rect.get_x() + rect.get_width() / 2.
                    y_pos = height # + max(values_matrix.max()) * 0.01
                    ax.text(x_pos, y_pos, f' {hit_items_pct:.1f}%',
                            ha='center', va='bottom', fontsize=7.5,
                            color=bar_color, rotation=90)
    
    # row < 3在非最后一列的柱子上方添加倍数标签
    if row < 3:
        for j in range(n):
            min_height = values_matrix[-1, j]
            for i in range(m):
                if i == m-1:
                    continue
                height = values_matrix[i][j]
                if min_height > 0:  # 避免除以零
                    ratio = height / min_height
                    x_pos = x[j] - offset + i * bar_spacing # + bar_spacing / 2.
                    y_pos = height # + max(values_matrix.max()) * 0.02
                    bar_color = get_color_bool(apporaches[i], 2, m < len(apporaches))
                    ax.text(x_pos, y_pos, f' {ratio:.2f}x',
                            ha='center', va='bottom', fontsize=7.5,
                            color=bar_color, rotation=90)
    
    # 设置坐标轴
    # ax.set_xlabel(x_label, fontsize=12)
    # ax.set_ylabel(y_label, fontsize=12)
    # ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    if row == 3:
        ax.set_xticklabels([f'QPS {qps_list[n][i]}' for i in range(n)])
    
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


qps_list2 = {
    1: [0.2, 0.25, 0.3],
    2: [0.2, 0.25, 0.3],
    3: [0.2, 0.25, 0.3],
}
def plot_grouped_bar_chart_tracea(ax: plt.Axes, 
                          data: Union[Dict, List, np.ndarray, Tuple],
                          row: int,
                          col: int,
                          x_label: str = "X轴",
                          y_label: str = "Y轴",
                          title: str = "分组柱状图",
                          bar_width: float = 0.7,
                          colors: List[str] = None,
                          legend_title: str = "类别",
                          show_values: bool = False,
                          value_format: str = "{:.1f}"):
    
    entry_num = 3
    values_matrix = np.zeros((len(apporaches) - 1, entry_num))
    values_matrix2 = np.zeros_like(values_matrix)
    
    # print(data)
    
    for (apporach, vals) in data.items():
        for i, qps in enumerate(qps_list2[entry_num]):
            try:
                entry = vals[qps]
            except KeyError:
                print(f"[WARN] 缺少数据: approach={apporach}, qps={qps}. vals={vals.keys()}")
                continue
            
            if row == 1:  # TTFT
                values_matrix[apporach_map[apporach]][i] = entry['ttft']
            elif row == 2:  # TPOT
                values_matrix[apporach_map[apporach]][i] = entry['tpot'] * 1000  # 转换为毫秒
            elif row == 3:  # Cache Hit Rate
                values_matrix[apporach_map[apporach]][i] = entry['hit_rate'] * 100  # 转换为百分比
                values_matrix2[apporach_map[apporach]][i] = entry['hit_items'] / 2100 * 100

    # print(f"Values matrix:\n{values_matrix}")
    # print(f"Values matrix2:\n{values_matrix2}")
            
    m, n = values_matrix.shape  # m: 类别数, n: x轴点数
    
    # 计算柱子位置
    x = np.arange(1, n + 1)  # x轴坐标从1开始
    bar_spacing = bar_width / m  # 每个柱子的宽度
    offset = bar_width / 2 - bar_spacing / 2  # 初始偏移量
    
    # 确定要高亮的类别
    if row < 3:
        # 找最小值所在的类别
        flat_idx = np.argmin(values_matrix)
        highlight_idx = flat_idx // n
    else:
        # 找最大值所在的类别
        flat_idx = np.argmax(values_matrix)
        highlight_idx = flat_idx // n
    
    highlight_map = np.zeros_like(values_matrix, dtype=bool)
    # 每一列最大值所在位置应该被高亮
    for j in range(n):
        col_vals = values_matrix[:, j]
        if row < 3:
            max_idx = np.argmin(col_vals)
        else:
            max_idx = np.argmax(col_vals)
        highlight_map[max_idx, j] = True
    
    bars = []  # 存储柱子对象
    
    # 绘制每个类别的柱子
    for i in range(m):
        # 计算当前类别柱子的x位置
        x_positions = x - offset + i * bar_spacing
        
        bars_this = []
        
        for j in range(n):
            # 根据是否为高亮类别设置hatch
            # hatch = '////' if highlight_map[i, j] else None
            hatch = '////' if highlight_map[i, j] else None
        
        
        # 绘制柱子
            if row == 3:
                bar_container = ax.bar(x_positions[j], values_matrix2[i][j], 
                                width=bar_spacing * 0.9,  # 稍微缩小柱子宽度，留出间距
                                color=get_color_bool(apporaches[i], 3, m < len(apporaches)),
                                edgecolor='white',
                                # hatchcolor='white',
                                linewidth=0.0)
                ax.bar(x_positions[j], values_matrix[i][j], 
                                    width=bar_spacing * 0.9,  # 稍微缩小柱子宽度，留出间距
                                    color=get_color_bool(apporaches[i], 1, m < len(apporaches)),
                                    edgecolor='gray',
                                    # hatchcolor='white',
                                    linewidth=0.3,
                                    hatch=hatch)
            else:
                bar_container = ax.bar(x_positions[j], values_matrix[i][j], 
                                    width=bar_spacing * 0.9,  # 稍微缩小柱子宽度，留出间距
                                    color=get_color_bool(apporaches[i], 1, m < len(apporaches)),
                                    edgecolor='white',
                                    # hatchcolor='white',
                                    linewidth=0.,
                                    hatch=hatch)
            # bar_container.set_hatch_color('white')
            bars.append(bar_container)
            bars_this.append(bar_container)
        
        # 显示数值
        if show_values:
            for rect, val in zip(bar_container, values_matrix[i]):
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height + max(values_matrix.max())*0.01,
                       value_format.format(val),
                       ha='center', va='bottom', fontsize=8)
        
        if row == 3:
            # 在命中率柱子上方添加命中项百分比标签
            bar_color = get_color_bool(apporaches[i], 2, m < 5)
            # print(f"len of bar_container: {len(bar_container)}, values_matrix2[i]: {values_matrix2[i]}")
            for j, rects in enumerate(bars_this):
                rect = rects.patches[0]  # 获取当前柱子对象
                hit_items_pct = values_matrix2[i][j]
                if hit_items_pct < 2.5:
                    height = rect.get_height()
                    x_pos = rect.get_x() + rect.get_width() / 2.
                    y_pos = height # + max(values_matrix.max()) * 0.01
                    ax.text(x_pos, y_pos, f' {hit_items_pct:.1f}%',
                            ha='center', va='bottom', fontsize=7.5,
                            color=bar_color, rotation=90)
    
    # row < 3在非最后一列的柱子上方添加倍数标签
    if row < 3:
        for j in range(n):
            min_height = values_matrix[-1, j]
            for i in range(m):
                if i == m-1:
                    continue
                height = values_matrix[i][j]
                if min_height > 0:  # 避免除以零
                    ratio = height / min_height
                    x_pos = x[j] - offset + i * bar_spacing # + bar_spacing / 2.
                    y_pos = height # + max(values_matrix.max()) * 0.02
                    bar_color = get_color_bool(apporaches[i], 2, m < len(apporaches))
                    ax.text(x_pos, y_pos, f' {ratio:.2f}x',
                            ha='center', va='bottom', fontsize=7.5,
                            color=bar_color, rotation=90)
    
    # 设置坐标轴
    # ax.set_xlabel(x_label, fontsize=12)
    # ax.set_ylabel(y_label, fontsize=12)
    # ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    if row == 3:
        ax.set_xticklabels([f'QPS {qps_list2[n][i]}' for i in range(n)])
        # set xticklabels closer to the x-axis
        ax.tick_params(axis='x', pad=0.5)
    
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