"""
对曲线进行分割，得到分段点
"""
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sensor import SAMPLE_RATE, SUPPORTED_SAMPLE_TYPES, add_noise, generate_sample
from scipy.signal import savgol_filter, find_peaks
import numpy as np

SEGMENT_POINT_1_THRESHOLD = 30
END_BLACKOUT_THRESHOLD = 0.2  # 计算分界点2时屏蔽最后X秒的数据，因为剧烈波动会干扰算法


def get_d(s, smooth=True, show_plt=False, name=""):
    """其实就是算斜率，具体请见论文"""
    x, y = s
    if smooth:
        """
        平滑滤波
        window_length：窗口长度，该值需为正奇整数
        k值：polyorder为对窗口内的数据点进行k阶多项式拟合，k的值需要小于window_length
        """
        y = savgol_filter(y, window_length=7, polyorder=3)
        y = savgol_filter(y, window_length=5, polyorder=1)
    if show_plt:  # debug usage
        plt.plot(x, y, label='original values')
        plt.plot(x, y, label="curve after filtering")
        plt.legend(loc='best')
        plt.title(f"{name} input")
        plt.show()
    assert len(x) > 2  # 算法要求至少需要2个点
    result = []
    for i in range(len(x)-1):  # 计算曲线的斜率
        t = (y[i+1]-y[i])/(x[i+1]-x[i])
        result.append(t)
    assert len(result) == len(x)-1  # 斜率数组的个数比点的个数少1
    if show_plt:  # debug usage
        draw_line(x, result+[0], title=f"{name} output")
    return x, result+[0]  # 返回斜率


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
    if threshold == 0:  # 递归中止条件，山高度阈值为0还找不到分段点，说明分段点不存在
        print("segmentation point 1 not found")
        return None, None
    if len(peak_idx) < 2:  # 找到的点不够，说明阈值太高，降低阈值再找
        threshold -= 1  # 降低“自适应阈值”
        print("applying adaptive threshhold: ", threshold)
        return find_segmentation_point_1(x, y, threshold)
    #print("peak_point_available: ", np.array(x)[peak_idx])
    index = peak_idx[1]  # 点的索引
    result = x[index]  # 点的x值（时间）
    #print("segmentation point 1: ", result)
    return index, result


def find_segmentation_point_2(x, y, segmentation_point_1_index):
    """寻找第二个分段点（between stage 2 and stage 3）"""
    end_blackout_length = round(
        SAMPLE_RATE*END_BLACKOUT_THRESHOLD)  # 屏蔽最后一段时间的数据，因为剧烈波动会干扰算法

    x, y = x[segmentation_point_1_index:-end_blackout_length],\
        y[segmentation_point_1_index:-end_blackout_length]  # 切掉stage 1和最后屏蔽的数据
    peak_idx, properties = find_peaks(y, prominence=0)  # 寻找峰值
    prominences = properties["prominences"]  # 峰值的详细参数
    if len(peak_idx) == 0 or len(prominences) == 0:  # 没有找到峰值，说明分段点不存在
        print("segmentation point 2 not found")
        return None, None
    #print("peak_point_available: ", np.array(x)[peak_idx])
    index = np.argmax(prominences)  # 找到最大的峰值
    result = x[peak_idx[index]]  # 峰值的x值（时间）
    #print("segmentation point 2: ", result)
    return index, result


def draw_line(x=None, y=None, title="", y_label="", is_dot=False):
    """绘制曲线（debug usage）"""
    assert y is not None
    if x is None:  # 如果没有x值，就用y值的索引作为x值
        x = [i for i in range(len(y))]
    plt.plot(x, y, "o" if is_dot else "b")
    plt.title(title)
    plt.xlabel("Time(s)")
    plt.ylabel(y_label)
    plt.show()


def calc_segmentation_points_single_series(series, name="", show_plt=False):
    """计算单条曲线的分段点"""
    x, y = series

    d1_result = get_d(series, smooth=True, show_plt=False,
                      name=f"{name} d1")  # 计算一阶导数
    d2_result = get_d(d1_result, smooth=True,
                      show_plt=False, name=f"{name} d2")  # 计算二阶导数
    segmentation_point_1_index, segmentation_point_1_x = find_segmentation_point_1(
        *d2_result)  # 寻找第一个分段点
    _, segmentation_point_2_x = find_segmentation_point_2(
        *d2_result, segmentation_point_1_index)  # 寻找第二个分段点
    if show_plt:  # debug usage
        fig = plt.figure(dpi=150, figsize=(9, 2))
        ax1 = fig.subplots()
        ax2 = ax1.twinx()  # 生成第二个y轴
        #ax2.plot(*d1_result, label="d1")
        ax2.plot(*d2_result, label="d2", color="red",
                 linewidth=1, alpha=0.2)
        ax1.plot(x, y, label="y", color="blue")
        # 画竖线
        if segmentation_point_1_x is not None:
            plt.axvline(x=segmentation_point_1_x, color='r', linestyle='--')
        if segmentation_point_2_x is not None:
            plt.axvline(x=segmentation_point_2_x, color='r', linestyle='--')
        plt.title(f"{name} final result")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2, loc='best')  # 显示图例
        plt.show()

    return segmentation_point_1_x, segmentation_point_2_x


def calc_segmentation_points(sample):
    """计算整个样本（4条线）的分段点"""
    result = {}
    for name, series in sample.items():  # 遍历每条曲线
        if name == "power":  # power曲线不作分段依据，因为感觉会起反作用
            continue
        result[name] = calc_segmentation_points_single_series(
            series, name=name, show_plt=False)  # 计算分段点
    print(result)
    # 做了一个融合，不同曲线算出的分段点可能不同，因此需要取最佳的分段点
    pt1, pt2 = [i[0] for i in result.values()], [i[1] for i in result.values()]
    # pt1和pt2中出现次数最多的值
    #final_result = max(set(pt1), key=pt1.count), max(set(pt2), key=pt2.count)
    # 求平均值
    pt1, pt2 = [i for i in pt1 if i is not None], [
        i for i in pt2 if i is not None]
    final_result = np.mean(pt1) if pt1 else None, np.mean(pt2) if pt2 else None
    # 特殊情况：如果第二个分段点小于等于第一个分段点，丢弃
    if final_result[0] and final_result[1] and final_result[1] <= final_result[0]:
        final_result = final_result[0], None
    print("segmentation final result: ", final_result)
    return final_result


if __name__ == "__main__":
    # calc_segmentation_points(generate_sample())

    for type in SUPPORTED_SAMPLE_TYPES:
        sample = generate_sample(type)
        print(sample.keys())
        name = "A"
        series = sample[name]
        calc_segmentation_points_single_series(
            series, name=f"{type} {name}", show_plt=True)
