from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sensor.config import SUPPORTED_SAMPLE_TYPES


def parse_predict_result(result):
    """解析预测结果"""
    result_pretty = [round(i, 2) for i in result.tolist()]
    result_pretty = dict(zip(SUPPORTED_SAMPLE_TYPES, result_pretty))  # 让输出更美观
    return result_pretty


def get_label_from_result_pretty(result_pretty):
    """从解析后的预测结果中获取标签"""
    return max(result_pretty, key=result_pretty.get)


def show_confusion_matrix(cm, labels=SUPPORTED_SAMPLE_TYPES):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot()
    plt.show()
