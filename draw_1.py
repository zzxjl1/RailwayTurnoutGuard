"""
SUPPORTED_SAMPLE_TYPES = ["normal", "H1", "H2", "H3",
                          "H4", "H5", "H6", "F1", "F2", "F3", "F4", "F5"]
4行3列子图组成大图，每个子图画出3条曲线，横轴为时间，纵轴为电流
"""

import matplotlib.pyplot as plt
import numpy as np
from sensor.simulate import generate_sample
from sensor.config import SUPPORTED_SAMPLE_TYPES

# 设置中文字体为宋体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def plot_samples():
    """
    绘制4x3的子图，展示不同类型样本的三相电流曲线
    """
    fig, axes = plt.subplots(4, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    # 为每种类型生成一个样本
    for i, sample_type in enumerate(SUPPORTED_SAMPLE_TYPES):
        # 生成单个样本数据
        sample, segmentations = generate_sample(sample_type)
        
        # 获取三相电流数据和对应的时间序列
        phases = ['A', 'B', 'C']
        
        # 在对应的子图中绘制三相电流
        ax = axes[i]
        for phase in phases:
            time, current = sample[phase]
            ax.plot(time, current, label=f'{phase} 相')
        
        # 设置子图标签
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('电流 (A)')
        ax.legend(loc='upper right')  # 强制legend在右上角
        ax.grid(True)
        
        # 设置x轴范围从0开始
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=5)  # y轴从0开始
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))  # 设置y轴刻度间隔为1
        
        # 在x轴标签下方添加子图标题，位置下移并添加标号
        subplot_label = chr(97 + i) # 将0,1,2...转换为a,b,c...
        ax.text(0.47, -0.5, f'({subplot_label}) {sample_type}类型样本', 
                horizontalalignment='center', 
                transform=ax.transAxes,
                fontsize=10)
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 进一步调整子图间距，减小上下间距
    plt.subplots_adjust(hspace=0.7)  # 减小垂直间距
    
    # 保存图片
    plt.savefig('samples.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_samples()