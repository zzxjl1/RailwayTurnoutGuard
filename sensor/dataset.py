import random
import numpy as np


try:
    from sensor.config import SUPPORTED_SAMPLE_TYPES
    from sensor.simulate import generate_sample
    from sensor.utils import find_nearest
except:
    from config import SUPPORTED_SAMPLE_TYPES
    from simulate import generate_sample
    from utils import find_nearest


def parse_time_series(time_series, time_series_length, pooling_factor_per_time_series):
    # 超长的截断，短的补0, 之后再降采样
    if len(time_series) > time_series_length:
        result = np.array(
            time_series[:time_series_length])
    else:
        result = np.pad(
            time_series, (0, time_series_length - len(time_series)), 'constant')
    result = result[::pooling_factor_per_time_series]
    return result


def parse_sample(sample, segmentations, time_series_length, pooling_factor_per_time_series, series_to_encode):
    time_series = []
    seg_index = []
    for name in series_to_encode:
        x, y = sample[name][0], sample[name][1]
        result = parse_time_series(
            y, time_series_length, pooling_factor_per_time_series)
        x = parse_time_series(x, time_series_length,
                              pooling_factor_per_time_series)
        assert len(x) == len(result)
        time_series.append(result)
        if segmentations is not None:
            seg_index = [find_nearest(x, seg) for seg in segmentations]
        else:
            seg_index = None
        # print(seg_index)

    result = np.array(time_series)
    return result, seg_index


def get_sample(type):
    """生成一个样本数组,shape为(channels, time_series_length//pooling_factor_per_time_series)"""
    if type is None:
        type = random.choice(SUPPORTED_SAMPLE_TYPES)
    sample, segmentations = generate_sample(type)


def generate_dataset(dataset_length, time_series_length, type=None, pooling_factor_per_time_series=1, series_to_encode=["A", "B", "C"]):
    """生成数据集"""
    x, seg_indexs = [], []
    for _ in range(dataset_length):
        if type is None:
            type = random.choice(SUPPORTED_SAMPLE_TYPES)
        sample, segmentations = generate_sample(type)

        time_series, seg_index = parse_sample(sample,
                                              segmentations,
                                              time_series_length,
                                              pooling_factor_per_time_series,
                                              series_to_encode)
        x.append(time_series)
        seg_indexs.append(seg_index)
    return np.array(x), seg_indexs


if __name__ == "__main__":
    BATCH_SIZE = 10
    TIME_SERIES_LENGTH = 100
    t, seg_index = generate_dataset(BATCH_SIZE, TIME_SERIES_LENGTH, type=None,
                                    pooling_factor_per_time_series=2, series_to_encode=["A", "B", "C"])
    print(t.shape, seg_index)
