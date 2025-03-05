import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和全局字体大小
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.rcParams['font.size'] = 12  # 设置全局默认字体大小
plt.rcParams['axes.labelsize'] = 14  # 设置坐标轴标签字体大小
plt.rcParams['axes.titlesize'] = 14  # 设置标题字体大小
plt.rcParams['xtick.labelsize'] = 12  # 设置x轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 12  # 设置y轴刻度标签字体大小
plt.rcParams['legend.fontsize'] = 12  # 设置图例字体大小
plt.rcParams['hatch.linewidth'] = 0.5  # 设置填充线的粗细

# 数据准备
methods = ['集成子模型\n(串行执行)', '集成子模型\n(并行执行)', 
          '神经元级并行', '本文方案']

# 计算时间数据 (ms)
compute_min = [718.31, 418.47, 304.98, 317.59]
compute_avg = [849.67, 548.83, 427.5, 459.01]
compute_max = [930.95, 895.82, 517.19, 583.45]

# 传输时间数据 (ms)
transfer_min = [4.75, 19.32, 72.73, 8.61]
transfer_avg = [5.47, 23.4, 81.62, 10.29]
transfer_max = [8.58, 38.17, 90.59, 13.84]

def create_bar_plot(ax, min_data, avg_data, max_data, title=''):
    """创建柱状图的通用函数"""
    x = np.arange(len(methods))
    width = 0.25
    
    # 使用适中密度的填充图案
    ax.bar(x - width, min_data, width, label='最小值', 
           color='white', edgecolor='#1f77b4', hatch='///')
    ax.bar(x, avg_data, width, label='平均值',
           color='white', edgecolor='#ff7f0e', hatch='\\\\\\')
    ax.bar(x + width, max_data, width, label='最大值',
           color='white', edgecolor='#2ca02c', hatch='xxx')
    
    ax.set_ylabel('时间开销 (ms)', fontsize=14)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)  # 降低网格线的显示强度
    
    # 添加数值标签，调整字体大小
    for i in range(len(methods)):
        ax.text(i - width, min_data[i], f'{min_data[i]}', ha='center', va='bottom', fontsize=11)
        ax.text(i, avg_data[i], f'{avg_data[i]}', ha='center', va='bottom', fontsize=11)
        ax.text(i + width, max_data[i], f'{max_data[i]}', ha='center', va='bottom', fontsize=11)

# 创建计算时间图
plt.figure(figsize=(9, 4))
ax1 = plt.gca()
create_bar_plot(ax1, compute_min, compute_avg, compute_max)
plt.tight_layout()
plt.savefig('compute_time_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 创建传输时间图
plt.figure(figsize=(9, 4))
ax2 = plt.gca()
create_bar_plot(ax2, transfer_min, transfer_avg, transfer_max)
plt.tight_layout()
plt.savefig('transfer_time_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

