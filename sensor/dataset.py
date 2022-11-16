import numpy as np


try:
    from sensor.simulate import generate_sample
    from sensor.utils import find_nearest
except:
    from simulate import generate_sample
    from utils import find_nearest


def parse(time_series, time_series_length):
    # 超长的截断，短的补0
    if len(time_series) > time_series_length:
        return np.array(
            time_series[:time_series_length])
    else:
        return np.pad(
            time_series, (0, time_series_length - len(time_series)), 'constant')


def get_sample(time_series_length, type, pooling_factor_per_time_series, series_to_encode):
    """获取拼接后的时间序列，比如Phase A, B, C连在一起，这样做是为了输入模型中"""
    temp, segmentations = generate_sample(
        type=type if type else np.random.choice(series_to_encode))

    time_series = []
    seg_index = []
    for name in series_to_encode:
        x, y = temp[name][0], temp[name][1]
        result = parse(y, time_series_length)
        result = result[::pooling_factor_per_time_series]
        x = x[::pooling_factor_per_time_series]
        time_series.append(result)

        seg_index = [find_nearest(x, seg) for seg in segmentations]
        # print(seg_index)

    result = np.array(time_series)
    return result, seg_index


def generate_dataset(dataset_length, time_series_length, type=None, pooling_factor_per_time_series=1, series_to_encode=["A", "B", "C"]):
    """生成数据集"""
    x, seg_indexs = [], []
    for _ in range(dataset_length):
        time_series, seg_index = get_sample(
            time_series_length, type, pooling_factor_per_time_series, series_to_encode)
        x.append(time_series)
        seg_indexs.append(seg_index)
    return np.array(x), seg_indexs


if __name__ == "__main__":
    BATCH_SIZE = 10
    TIME_SERIES_LENGTH = 100
    t, seg_index = generate_dataset(BATCH_SIZE, TIME_SERIES_LENGTH, type="normal",
                                    pooling_factor_per_time_series=2, series_to_encode=["A", "B", "C"])
    print(t.shape, seg_index)
