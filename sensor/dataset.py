import numpy as np
import random

from scipy import signal

from sensor.utils import shuffle


try:
    from sensor.config import SUPPORTED_SAMPLE_TYPES, USE_SIMULATED_DATA
    from sensor.simulate import generate_sample
    from sensor.utils import find_nearest
    from sensor.real_world import get_all_samples
except:
    from config import SUPPORTED_SAMPLE_TYPES, USE_SIMULATED_DATA
    from simulate import generate_sample
    from utils import find_nearest
    from real_world import get_all_samples


def parse_time_series(time_series, time_series_length, pooling_factor_per_time_series):
    # 超长的截断，短的补0, 之后再降采样
    if len(time_series) > time_series_length:
        result = np.array(time_series[:time_series_length])
    else:
        result = np.pad(
            time_series, (0, time_series_length - len(time_series)), "constant"
        )
    # result = result[::pooling_factor_per_time_series]
    result = signal.decimate(result, pooling_factor_per_time_series)
    return result


def parse_sample(
    sample,
    segmentations,
    time_series_length,
    pooling_factor_per_time_series,
    series_to_encode,
):
    time_series = []
    seg_index = []
    for name in series_to_encode:
        x, y = sample[name][0], sample[name][1]
        result = parse_time_series(
            y, time_series_length, pooling_factor_per_time_series
        )
        x = parse_time_series(x, time_series_length, pooling_factor_per_time_series)
        assert len(x) == len(result)
        time_series.append(result)
        if segmentations is not None:
            seg_index = [find_nearest(x, seg) for seg in segmentations]
        else:
            seg_index = None
        # print(seg_index)

    result = np.array(time_series)
    return result, seg_index


def get_sample_fake(type):
    return generate_sample(type)


def get_sample_real(type):
    from segmentation import calc_segmentation_points

    samples, types = get_all_samples(type_list=[type])
    sample = random.choice(samples)
    segmentations = calc_segmentation_points(sample)
    return sample, segmentations


def generate_dataset_real(
    dataset_length,
    time_series_length,
    sample_type=None,
    pooling_factor_per_time_series=1,
    series_to_encode=["A", "B", "C"],
    no_segmentation=False,
):
    """生成数据集"""

    from segmentation import calc_segmentation_points

    x, seg_indexs, types = [], [], []

    if sample_type is None:
        samples, types = get_all_samples()
    elif isinstance(sample_type, list):
        samples, types = get_all_samples(type_list=sample_type)
    elif isinstance(sample_type, str):
        samples, types = get_all_samples(type_list=[sample_type])
    else:
        raise ValueError()

    if dataset_length != -1:
        samples, types = samples[:dataset_length], types[:dataset_length]

    for sample in samples:
        array_sample, seg_index = parse_sample(
            sample,
            calc_segmentation_points(sample) if not no_segmentation else None,
            time_series_length,
            pooling_factor_per_time_series,
            series_to_encode,
        )

        x.append(array_sample)
        seg_indexs.append(seg_index)

    x, seg_indexs, types = shuffle(x, seg_indexs, types)
    return np.array(x), seg_indexs, types


def generate_dataset_fake(
    dataset_length,
    time_series_length,
    sample_type=None,
    pooling_factor_per_time_series=1,
    series_to_encode=["A", "B", "C"],
):
    """生成数据集"""
    x, seg_indexs, types = [], [], []
    for _ in range(dataset_length):
        if sample_type is None:
            type = random.choice(SUPPORTED_SAMPLE_TYPES)
        elif isinstance(sample_type, list):
            type = random.choice(sample_type)
        else:
            type = sample_type
        sample, segmentations = get_sample_fake(type)
        array_sample, seg_index = parse_sample(
            sample,
            segmentations,
            time_series_length,
            pooling_factor_per_time_series,
            series_to_encode,
        )

        x.append(array_sample)
        seg_indexs.append(seg_index)
        types.append(type)
    return np.array(x), seg_indexs, types


if __name__ == "__main__":
    TIME_SERIES_LENGTH = 100
    BATCH_SIZE = 32
    t, seg_indexs, types = generate_dataset_fake(
        BATCH_SIZE,
        TIME_SERIES_LENGTH,
        sample_type=None,
        pooling_factor_per_time_series=2,
        series_to_encode=["A", "B", "C"],
    )
    print(t.shape, seg_indexs, types)
