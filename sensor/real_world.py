"""
读取excel表格中的现实世界数据并打包为sample
"""
import numpy as np
try:
    from sensor.utils import show_sample, interpolate
except:
    from utils import show_sample, interpolate
import pandas as pd
from enum import Enum


column_names = ["timestamp", "direction", "data",
                "point_count", "current_type", "curve_type", "turnout_name"]  # 列名
df = pd.read_excel("./sensor/turnoutActionCurve.xlsx")  # 读取excel文件


class CurveType(Enum):
    A = 1
    B = 2
    C = 3
    power = 4


class CurrentType(Enum):
    AC = 0
    DC = 1


def read_row(i):
    # 获取第i行的数据
    row_data = df.iloc[i, :]
    # 映射到column_names
    result = dict(zip(column_names, row_data.values))
    return result


seq = ["A", "B", "C", "power"]  # 曲线顺序


def validate(i):
    for j in range(4):  # 遍历四个曲线
        t = read_row(i+j)  # 获取第i+j行的数据
        if t["curve_type"] != CurveType[seq[j]].value:  # 顺序不匹配
            return False
        return True


def polish_data(data, point_count, type):
    POINT_INTERVAL = 40/1000  # 40ms
    DURATION = POINT_INTERVAL*point_count  # 时长

    x = np.linspace(0, DURATION, point_count)  # 生成x轴数据
    y = data.split(",")  # 生成y轴数据
    if type in ["A", "B", "C"]:
        y = [float(i)/100 for i in y]  # y全部元素除以100
    assert len(x) == len(y)  # x和y长度相等

    return interpolate(x, y)  # 插值


def parse(i):
    result = {}
    for j in range(4):  # 遍历四个曲线
        t = read_row(i+j)  # 获取第i+j行的数据
        type = CurveType(t["curve_type"]).name  # 曲线类型
        result[type] = polish_data(
            t["data"], t["point_count"], type)  # 解析数据
    # print(result)
    # show_sample(result)
    return result


def get_all_samples():
    result = []
    row = df.shape[0]  # 总行数
    i = 0
    while i < row:  # 遍历每一行

        """
        if t["current_type"] == CurrentType.DC.value: # 过滤直流转辙机
            i += 1
            continue
        """

        if not validate(i):  # 过滤非法数据
            i += 1
            print(f"line {i} validate failed")
            continue

        print(f"line {i} parsed")
        result.append(parse(i))  # 解析数据后添加到result
        i += 4

    print(len(result))
    return result


if __name__ == "__main__":
    import os
    import sys

    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parentdir)

    from gru_score import GRUScore
    from segmentation import calc_segmentation_points
    from auto_encoder import model_input_parse, predict, visualize_prediction_result

    for sample in get_all_samples():
        show_sample(sample)
        calc_segmentation_points(sample, show_plt=True)  # 计算分割点

        model_input = model_input_parse(sample)
        results, losses = predict(model_input)
        visualize_prediction_result(model_input, results, losses)
