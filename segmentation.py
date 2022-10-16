from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sensor import SAMPLE_RATE, SUPPORTED_SAMPLE_TYPES, add_noise, generate_sample
from scipy.signal import savgol_filter, find_peaks
import numpy as np

SEGMENT_POINT_1_THRESHOLD = 30
END_BLACKOUT_THRESHOLD = 0.25  # 计算分界点2时屏蔽最后0.25秒的数据，因为剧烈波动会干扰算法


def get_d(s, smooth=True, show_plt=False, name=""):
    """
    平滑滤波
    window_length：窗口长度，该值需为正奇整数
    k值：polyorder为对窗口内的数据点进行k阶多项式拟合，k的值需要小于window_length
    """
    x, y = s
    if smooth:
        y = savgol_filter(y, window_length=7, polyorder=3)
        y = savgol_filter(y, window_length=5, polyorder=1)
    if show_plt:
        plt.plot(x, y, label='original values')
        plt.plot(x, y, label="curve after filtering")
        plt.legend(loc='best')
        plt.title(f"{name} input")
        plt.show()
    assert len(x) > 2
    result = []
    for i in range(len(x)-1):
        t = (y[i+1]-y[i])/(x[i+1]-x[i])
        result.append(t)
    assert len(result) == len(x)-1
    if show_plt:
        draw_line(x, result+[0], title=f"{name} output")
    return x, result+[0]


def remove_duplicate_points(points):
    """去除重复的分段点"""
    result = []
    for i in points:
        if i not in result:
            result.append(i)
    return result


def find_segmentation_point_1(x, y, threshold=SEGMENT_POINT_1_THRESHOLD):
    """寻找第一个分段点（between stage 1 and stage 2）"""
    peak_idx, _ = find_peaks(y, height=threshold)
    if threshold == 0:
        print("segmentation point 1 not found")
        return None, None
    if len(peak_idx) < 2:
        threshold -= 1
        print("applying adaptive threshhold: ", threshold)
        return find_segmentation_point_1(x, y, threshold)
    print("peak_point_available: ", np.array(x)[peak_idx])
    index = peak_idx[1]
    result = x[index]
    print("segmentation point 1: ", result)
    return index, result


def find_segmentation_point_2(x, y, segmentation_point_1_index, type="normal"):
    """寻找第二个分段点（between stage 2 and stage 3）"""
    end_blackout_length = round(
        SAMPLE_RATE*END_BLACKOUT_THRESHOLD)

    if type == "F4":
        x, y = x[segmentation_point_1_index:],\
            y[segmentation_point_1_index:]
    else:
        x, y = x[segmentation_point_1_index:-end_blackout_length],\
            y[segmentation_point_1_index:-end_blackout_length]
    peak_idx, properties = find_peaks(y, prominence=0)
    prominences = properties["prominences"]
    if len(peak_idx) == 0 or len(prominences) == 0:
        print("segmentation point 2 not found")
        return None, None
    print("peak_point_available: ", np.array(x)[peak_idx])
    index = np.argmax(prominences)
    result = x[peak_idx[index]]
    print("segmentation point 2: ", result)
    return index, result


def draw_line(x=None, y=None, title="", y_label="", is_dot=False):
    """绘制曲线（debug usage）"""
    assert y is not None
    if x is None:
        x = [i for i in range(len(y))]
    plt.plot(x, y, "o" if is_dot else "b")
    plt.title(title)
    plt.xlabel("Time(s)")
    plt.ylabel(y_label)
    plt.show()


def calc_segmentation_points(series, type="normal", show_plt=False):

    x, y = series

    d1_result = get_d(series, smooth=True, show_plt=False, name=f"{type} d1")
    d2_result = get_d(d1_result, smooth=True,
                      show_plt=False, name=f"{type} d2")
    segmentation_point_1_index, segmentation_point_1_x = find_segmentation_point_1(
        *d2_result)
    _, segmentation_point_2_x = find_segmentation_point_2(
        *d2_result, segmentation_point_1_index, type)
    if show_plt:
        fig = plt.figure(dpi=150, figsize=(9, 2))
        ax1 = fig.subplots()
        ax2 = ax1.twinx()
        #ax2.plot(*d1_result, label="d1")
        ax2.plot(*d2_result, label="d2", color="red",
                 linewidth=1, alpha=0.2)
        ax1.plot(x, y, label="y", color="blue")
        # 画竖线
        if segmentation_point_1_x is not None:
            plt.axvline(x=segmentation_point_1_x, color='r', linestyle='--')
        if segmentation_point_2_x is not None:
            plt.axvline(x=segmentation_point_2_x, color='r', linestyle='--')
        plt.title(f"{type} final result")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2, loc='best')
        plt.show()

    return segmentation_point_1_x, segmentation_point_2_x


if __name__ == "__main__":
    for type in SUPPORTED_SAMPLE_TYPES:
        sample = generate_sample(type)
        print(sample.keys())
        name = "A"
        series = sample[name]
        calc_segmentation_points(series, type, show_plt=True)
