"""
依据分段结果，使用RNN逐段提取特征后进行分类
这样避免漏掉手动特征工程无法涵盖的特征
之后会和dnn、ae的结果进行融合
"""
from segmentation import calc_segmentation_points
from sensor import get_sample, SUPPORTED_SAMPLE_TYPES


def generate_dataset():
    pass
