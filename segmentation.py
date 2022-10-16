
from cProfile import label
from math import exp
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sensor import add_noise, generate_sample
from scipy.signal import savgol_filter, find_peaks
import numpy as np

SEGMENT_POINT_1_THRESHOLD = 30


def get_d(s, smooth=True, show_plt=False, name=""):
    """
    平滑滤波
    window_length：窗口长度，该值需为正奇整数
    k值：polyorder为对窗口内的数据点进行k阶多项式拟合，k的值需要小于window_length
    """
    x, y = s

    y = savgol_filter(y, window_length=13, polyorder=3)
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
    print("peak_point_x: ", x[peak_idx])
    if len(peak_idx) < 2:
        threshold -= 1
        print("applying adaptive threshhold: ", threshold)
        return find_segmentation_point_1(x, y, threshold)
    index = peak_idx[1]
    result = x[index]
    print("segmentation point 1: ", result)
    return index, result


def find_segmentation_point_2(x, y, segmentation_point_1_index):
    """寻找第二个分段点（between stage 2 and stage 3）"""
    x, y = x[segmentation_point_1_index:], y[segmentation_point_1_index:]
    peak_idx, properties = find_peaks(y, prominence=0)
    #print("peak_point_x: ", x[peak_idx])
    index = np.argmax(properties["prominences"])
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


if __name__ == "__main__":

    sample = generate_sample("normal", show_plt=False)
    print(sample.keys())
    name = "A"
    series = sample[name]
    x, y = series

    d1_result = get_d(series, smooth=True, show_plt=False, name="d1")
    d2_result = get_d(d1_result, smooth=False, show_plt=False, name="d2")
    segmentation_point_1_index, segmentation_point_1_x = find_segmentation_point_1(
        *d2_result)
    _, segmentation_point_2_x = find_segmentation_point_2(
        *d2_result, segmentation_point_1_index)
    #plt.plot(*d2_result, label="d2")
    plt.plot(x, y, label="y")
    # 画竖线
    plt.axvline(x=segmentation_point_1_x, color='r', linestyle='--')
    plt.axvline(x=segmentation_point_2_x, color='r', linestyle='--')
    plt.title("final result")
    plt.show()
