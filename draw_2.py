"""
可视化二阶差分序列生成过程如图3-2所示，可以直观地观察到这些特征点与实际分段点的对应关系。图3-2中从上到下依次展示了原始电流序列、一阶差分 D^'、二阶差分 D^'' 结果。通过观察可以发现，在二阶差分序列D^''中出现了几个显著的峰值点，这些峰值反映了电流序列中变化率的突变。具体而言，当电流序列发生剧烈变化时（如启动、转换瞬间），一阶差分 D^' 会出现大幅跳变，这种跳变在二阶差分 D^'' 中表现为正负相间的尖峰。特别是在电流曲线转折点处（如 P_1 和 P_2 位置），由于变化率的突然改变，会在二阶差分中形成显著的峰值。
画图，1列n行，第一行是原始序列，第二行是一阶差分后的，，第三行是二阶，标出所有峰值，
"""

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
from sensor.simulate import generate_sample
from segmentation import get_d

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def plot_differential_analysis(sample_type="normal"):
    """绘制差分序列分析图"""
    # 生成样本数据
    sample, segmentations = generate_sample(sample_type)
    series = sample['A']  # 使用A相电流数据
    
    # 计算一阶和二阶差分
    d1_result = get_d(series, smooth=True)
    d2_result = get_d(d1_result, smooth=True)
    
    # 创建三个子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 5))
    
    # 获取最大时间值用于统一x轴范围
    max_time = max(series[0][-1], d1_result[0][-1], d2_result[0][-1])
    
    # 绘制原始序列
    ax1.plot(*series, label='原始电流序列')
    ax1.set_ylabel('电流 (A)', labelpad=10)
    ax1.grid(True)
    ax1.legend()
    ax1.set_xlim(0, max_time)
    
    # 绘制一阶差分
    ax2.plot(*d1_result, label='一阶差分序列')
    # 找出一阶差分的峰值
    peaks1, _ = find_peaks(d1_result[1], prominence=0.1)
    peaks1 = np.array(peaks1).astype(np.int64)
    time_array = np.array(d1_result[0])
    current_array = np.array(d1_result[1])
    ax2.plot(time_array[peaks1], current_array[peaks1], "rx", label='峰值点')
    ax2.set_ylabel('一阶差分值', labelpad=10)
    ax2.grid(True)
    ax2.legend()
    ax2.set_xlim(0, max_time)
    
    # 绘制二阶差分
    ax3.plot(*d2_result, label='二阶差分序列')
    # 找出二阶差分的峰值
    peaks2, _ = find_peaks(d2_result[1], prominence=0.1)
    peaks2 = np.array(peaks2).astype(np.int64)
    time_array = np.array(d2_result[0])
    current_array = np.array(d2_result[1])
    ax3.plot(time_array[peaks2], current_array[peaks2], "rx", label='峰值点')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('二阶差分值', labelpad=10)
    ax3.grid(True)
    ax3.legend()
    ax3.set_xlim(0, max_time)
    
    # 在所有子图中标注分段点
    legend_added = False  # 用于确保只添加一次图例
    for ax in [ax1, ax2, ax3]:
        for seg_point in segmentations:
            if seg_point is not None:
                label = '实际分段点' if not legend_added else None
                ax.axvline(x=seg_point, color='g', linestyle='--', 
                          label=label, alpha=0.5)
                legend_added = True
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(f'differential_analysis_{sample_type}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_all_types():
    """为所有类型的样本绘制差分分析图"""
    from sensor.config import SUPPORTED_SAMPLE_TYPES
    for sample_type in SUPPORTED_SAMPLE_TYPES:
        print(f"Processing {sample_type}...")
        plot_differential_analysis(sample_type)

if __name__ == "__main__":
    # 绘制单个类型的分析图
    plot_differential_analysis("H2")
    
    # 或者绘制所有类型的分析图
    # plot_all_types()