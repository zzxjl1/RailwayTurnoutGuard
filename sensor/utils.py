import random
import matplotlib.pyplot as plt
import numpy as np
try:
    from .config import SAMPLE_RATE
except:
    from config import SAMPLE_RATE
import scipy.interpolate


def random_float(a, b, n=2):
    """产生a和b之间随机小数,保留n位"""
    return round(random.uniform(a, b), n)


def tansform_to_plt(points):
    """将坐标点(x,y)转换为plt可用的格式(x),(y)'
    eg: [(9,0),(1,5),(2,4)] -> ([9,1,2],[0,5,4])
    """
    return list(zip(*points))


def find_nearest(array, value):
    """找到最近的点，返回索引"""
    if value is None:
        return None
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def generate_power_series(current_series, power_factor=0.8, show_plt=False):
    """
    产生瓦数曲线，采用直接计算的方式，需传入三项电流曲线
    P (kW) = I (Amps) × V (Volts) × PF(功率因数) × 1.732
    """
    # x, _ = current_series['A']
    # 取三相电流曲线最长的作为功率曲线x轴
    x = []
    for series in current_series.values():
        t, _ = series
        if len(t) > len(x):
            x = t
    length = len(x)
    result = np.zeros(length)
    for phase in ["A", "B", "C"]:
        for i in range(length):
            _, current = current_series[phase]
            result[i] += current[i]*220*power_factor * \
                1.732 if i < len(current) else 0
    if show_plt:
        plt.plot(x, result)
        plt.title("Power Series")
        plt.xlabel("Time(s)")
        plt.ylabel("Power(W)")
        plt.show()
    return x, result


def show_sample(result, type=""):
    fig = plt.figure(dpi=150, figsize=(9, 2))
    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    for phase in ["A", "B", "C"]:
        ax1.plot(*result[phase], label=f"Phase {phase}")
    ax2.plot(*result["power"], 'b--', label="Power")
    plt.title(f"Sample {type.capitalize()}")
    ax1.set_xlabel("Time(s)")
    ax1.set_ylabel("Current(A)")
    ax2.set_ylabel('Power(W)')

    ax1.set_ylim(bottom=0, top=5)
    ax2.set_ylim(bottom=0, top=5000)  # 限制y轴范围，避免图像过于密集

    plt.xlim(0, None)  # 设置x轴范围
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='best')
    plt.show()


def add_noise(x, y, noise_level=0.05, percentage=0.2):
    """加入抖动噪声"""
    if isinstance(noise_level, float):  # 如果为浮点数
        noice_range = (-noise_level, noise_level)
    elif isinstance(noise_level, tuple):  # 如果为元组
        if random.random() < 0.5:
            # (1,2) -> (-2,-1)
            noice_range = (-noise_level[1], -noise_level[0])
        else:
            noice_range = noise_level
    n = [random_float(*noice_range)*random.choice([1, -1])
         for _ in range(len(x))]
    for i in range(len(x)):
        if random.random() > percentage:  # 按概率加入噪声
            continue
        if y[i] == 0:  # 值为0的点不加噪声
            continue
        y[i] += n[i]  # 加入噪声
    return correct_curve(x, y)


def correct_curve(x, y):
    """修正曲线,小于0的值设为0"""
    return x, [0 if i < 0 else i for i in y]


def interpolate(x, y):
    """根据关键点插值到固定采样率"""
    interper = scipy.interpolate.interp1d(x, y, kind='linear')  # 线性插值
    time_elipsed = max(x)-min(x)  # 总时间
    x = np.linspace(min(x), max(x), round(
        time_elipsed*SAMPLE_RATE))  # 插值
    y = interper(x)

    return correct_curve(x, y)


def draw_line(x, y, title="", y_label=""):
    """绘制曲线（debug usage）"""
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Time(s)")
    plt.ylabel(y_label)
    plt.show()
