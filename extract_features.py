"""
从sample（共4条曲线）中提取特征
"""

import random
from segmentation import calc_segmentation_points, calc_segmentation_points_single_series
from sensor import SUPPORTED_SAMPLE_TYPES, find_nearest, generate_sample
from collections import OrderedDict
import matplotlib.pyplot as plt


def calc_features_per_stage(x, y, series_name, stage_name):
    """计算单个stage的特征"""

    result = OrderedDict()

    if x is not None and y is not None:
        result[f"{series_name}_{stage_name}_time_span"] = x[-1]-x[0]  # 时间跨度
        result[f"{series_name}_{stage_name}_max"] = max(y)  # 最大值
        result[f"{series_name}_{stage_name}_min"] = min(y)  # 最小值
        result[f"{series_name}_{stage_name}_mean"] = sum(y)/len(y)  # 平均值
        result[f"{series_name}_{stage_name}_median"] = sorted(y)[
            len(y)//2]  # 中位数
        result[f"{series_name}_{stage_name}_std"] = (sum([(i-result[f"{series_name}_{stage_name}_mean"])**2 for i in y]) /
                                                     len(y))**0.5  # Standard deviation
        result[f"{series_name}_{stage_name}_kurtosis"] = sum(
            [(i-result[f"{series_name}_{stage_name}_mean"])**4 for i in y]) / len(y)  # 峭度
        result[f"{series_name}_{stage_name}_rms"] = (
            sum([i**2 for i in y]) / len(y))**0.5  # 均方根

        if sum(y)/len(y) != 0:  # 防止除数为0
            result[f"{series_name}_{stage_name}_impluse_factor"] = max(
                y)/(sum(y)/len(y))  # Impulse factor
            result[f"{series_name}_{stage_name}_peak_factor"] = (
                max(y)-min(y))/(sum(y)/len(y))  # Peak factor

        else:
            result[f"{series_name}_{stage_name}_impluse_factor"] = 0
            result[f"{series_name}_{stage_name}_peak_factor"] = 0
        # nan值置0
        for k, v in result.items():
            if v != v:
                result[k] = 0
    else:
        result[f"{series_name}_{stage_name}_time_span"] = 0  # 同上
        result[f"{series_name}_{stage_name}_max"] = 0
        result[f"{series_name}_{stage_name}_min"] = 0
        result[f"{series_name}_{stage_name}_mean"] = 0
        result[f"{series_name}_{stage_name}_median"] = 0
        result[f"{series_name}_{stage_name}_std"] = 0
        result[f"{series_name}_{stage_name}_kurtosis"] = 0
        result[f"{series_name}_{stage_name}_rms"] = 0

        result[f"{series_name}_{stage_name}_impluse_factor"] = 0
        result[f"{series_name}_{stage_name}_peak_factor"] = 0

    assert len(result) == 10  # 断言确保10个特征
    return result


def calc_features(sample):
    """
    计算整个sample（4个曲线*3个stage）的特征
    （stage时间跨度、最大值、最小值、平均值、中位数、Standard deviation、Peak factor、Fluctuation factor）*4个曲线*3个stage
    """
    segmentation_points = calc_segmentation_points(sample)
    result = OrderedDict()
    for name, series in sample.items():  # 遍历4个曲线
        x, y = series
        total_time_elipsed = x[-1]-x[0]  # 总用时
        t = calc_features_single_series(x, y, segmentation_points, name)
        result.update(t)  # 合并
    result["total_time_elipsed"] = total_time_elipsed

    assert len(result) == 1+10*3*4  # 断言确保特征数
    return result


def calc_features_single_series(x, y, segmentation_points, series_name):
    """
    计算单个曲线的（见下方列出）种特征
    （stage时间跨度、最大值、最小值、平均值、中位数、Standard deviation、Peak factor、Fluctuation factor）*3个stage
    """
    features = OrderedDict()

    segmentation_points_count = len(
        [i for i in segmentation_points if i is not None])  # 计算划分点数

    x, y = map(lambda x: list(x), (x, y))

    if segmentation_points_count == 2:  # 2个划分点

        stage1_start, stage1_end = 0, find_nearest(x,
                                                   segmentation_points[0])  # 第一个stage的起始点和终止点索引
        stage1 = calc_features_per_stage(
            x[stage1_start:stage1_end], y[stage1_start:stage1_end], series_name, "stage1")  # 计算第一个stage的特征
        features.update(stage1)  # 合并

        stage2_start, stage2_end = stage1_end, find_nearest(x,
                                                            segmentation_points[1])  # 第二个stage的起始点和终止点索引
        stage2 = calc_features_per_stage(
            x[stage2_start:stage2_end], y[stage2_start:stage2_end], series_name, "stage2")  # 计算第二个stage的特征
        features.update(stage2)  # 合并

        stage3_start, stage3_end = stage2_end, len(x)  # 第三个stage的起始点和终止点索引
        stage3 = calc_features_per_stage(
            x[stage3_start:stage3_end], y[stage3_start:stage3_end], series_name, "stage3")  # 计算第三个stage的特征
        features.update(stage3)  # 合并

    elif segmentation_points_count == 1:  # 1个划分点

        assert segmentation_points[0] is not None
        stage1_start, stage1_end = 0, find_nearest(x, segmentation_points[0])
        stage1 = calc_features_per_stage(
            x[stage1_start:stage1_end], y[stage1_start:stage1_end], series_name, "stage1")
        features.update(stage1)

        stage2_start, stage2_end = stage1_end, len(x)
        stage2 = calc_features_per_stage(
            x[stage2_start:stage2_end], y[stage2_start:stage2_end], series_name, "stage2")
        features.update(stage2)

        stage3 = calc_features_per_stage(None, None, series_name, "stage3")
        features.update(stage3)

    elif segmentation_points_count == 0:  # 0个划分点

        stage1_start, stage1_end = 0, len(x)
        stage1 = calc_features_per_stage(
            x[stage1_start:stage1_end], y[stage1_start:stage1_end], series_name, "stage1")
        features.update(stage1)

        stage2 = calc_features_per_stage(None, None, series_name, "stage2")
        features.update(stage2)

        stage3 = calc_features_per_stage(None, None, series_name, "stage3")
        features.update(stage3)

    assert len(features) == 10*3
    return features


def test(type=None):
    if not type:
        type = random.choice(SUPPORTED_SAMPLE_TYPES)
    sample, _ = generate_sample(type)
    result = calc_features(sample)
    # print(result)
    return type, result


if __name__ == "__main__":
    print(test())

    results = {}
    for _ in range(100):
        type, features = test()
        for k, v in features.items():
            if k not in results:
                results[k] = [(v, type)]
            else:
                results[k].append((v, type))

    print(results)
    for name in results:
        t = {}
        for type in SUPPORTED_SAMPLE_TYPES:
            t[type] = [i[0] for i in results[name] if i[1] == type]
        results[name] = t

    print(results)
    count = 1
    for k, v in results.items():
        height, width = 13, 10
        #height, width = 10, 5
        if count > height*width:
            break
        plt.subplot(height, width, count)
        count += 1
        plt.title(k, fontsize=10)
        for type, data in v.items():
            plt.plot([i for i in range(len(data))], data, label=type)
    # plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
