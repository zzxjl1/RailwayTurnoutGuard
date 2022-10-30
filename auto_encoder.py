"""
降噪自编码器 Denoising Auto-Encoder
采用正常时间序列无监督训练，用于产生是否异常的置信度
该置信度会用于之后的分类，以降低假阳率
"""
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sensor import SAMPLE_RATE, SUPPORTED_SAMPLE_TYPES, generate_sample

POOLING_FACTOR_PER_TIME_SERIES = 5  # 每条时间序列的采样点数
TIME_SERIES_DURATION = 10  # 输入模型的时间序列时长为10s
TIME_SERIES_LENGTH = SAMPLE_RATE * \
    TIME_SERIES_DURATION//POOLING_FACTOR_PER_TIME_SERIES  # 时间序列长度
TRAINING_SET_LENGTH = 200  # 训练集长度
TESTING_SET_LENGTH = 50  # 测试集长度
SERIES_TO_ENCODE = ["A", "B", "C"]  # 参与训练和预测的序列，power暂时不用

LEARNING_RATE = 1e-3  # 学习率
BATCH_SIZE = 64  # 批大小
EPOCHS = 300  # 训练轮数

FILENAME = './models/auto_encoder.pth'  # 模型保存路径
FORCE_CPU = True  # 强制使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() and not FORCE_CPU
                      else 'cpu')
print('Using device:', DEVICE)

TOTAL_LENGTH = TIME_SERIES_LENGTH * len(SERIES_TO_ENCODE)  # 输入总长度
print("total input length:", TOTAL_LENGTH)

model = nn.Sequential(
    nn.Linear(TOTAL_LENGTH, round(TOTAL_LENGTH/5)),
    nn.Linear(round(TOTAL_LENGTH/5), TOTAL_LENGTH),
).to(DEVICE)  # 定义模型，很简单的AE,注意中间层的维度必须<<输入才有效

loss_func = nn.MSELoss()  # 均方误差
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # 优化器


def parse(time_series):
    # 降采样
    time_series = time_series[::POOLING_FACTOR_PER_TIME_SERIES]
    # 超长的截断，短的补0
    if len(time_series) > TIME_SERIES_LENGTH:
        return np.array(
            time_series[:TIME_SERIES_LENGTH])
    else:
        return np.pad(
            time_series, (0, TIME_SERIES_LENGTH - len(time_series)), 'constant')


def get_sample(type):
    """获取拼接后的时间序列，比如Phase A, B, C连在一起，这样做是为了输入模型中"""
    temp, _ = generate_sample(type=type)
    time_series = []
    for type in SERIES_TO_ENCODE:
        time_series += list(parse(temp[type][1]))  # concat操作
    return time_series


def generate_dataset():
    """生成数据集"""
    DATASET_LENGTH = TRAINING_SET_LENGTH + TESTING_SET_LENGTH  # 数据集总长度
    DATASET = []
    for _ in range(DATASET_LENGTH):
        time_series = get_sample("normal")  # 必须只用正常样本训练
        DATASET.append(time_series)
    return DATASET


def get_dataloader():
    temp = generate_dataset()
    DATASET = torch.tensor(np.array(temp), dtype=torch.float,
                           requires_grad=True).to(DEVICE)  # 转换为tensor
    print(DATASET.shape)
    train_ds = TensorDataset(DATASET[:TRAINING_SET_LENGTH])
    test_ds = TensorDataset(DATASET[TRAINING_SET_LENGTH:])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
    return train_dl, test_dl


def loss_batch(model, x, is_train):
    if is_train:
        y = x + torch.randn(x.shape) * 0.25  # 加入噪声
        result = model(y)  # 将加了噪声的数据输入模型
    else:
        result = model(x)
    loss = loss_func(result, x)  # 目标值为没加噪声的x
    loss.requires_grad_(True)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item(), len(x)


def train():
    train_dl, test_dl = get_dataloader()
    for epoch in range(EPOCHS):
        model.train()
        for x in train_dl:
            loss_batch(model, x[0], is_train=True)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, x[0], is_train=False) for x in test_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)

    torch.save(model, FILENAME)  # 保存模型


def predict(x):
    if not os.path.exists(FILENAME):
        train()
    model = torch.load(FILENAME)
    model.eval()
    with torch.no_grad():
        return model(x)


def draw(y_before, y_after, title=""):
    plt.plot(y_before)
    plt.plot(y_after)
    # x轴n等分，画竖线
    for i in range(len(SERIES_TO_ENCODE)+1):
        plt.axvline(x=TIME_SERIES_LENGTH*(i), color='r')
    plt.axvline(x=TIME_SERIES_LENGTH, color='r', linestyle='--')
    plt.title(title)
    plt.show()


def test(type="normal", show_plt=False):
    """生成一个正常样本，并进行正向传播，如果输出与输入相似，则说明模型训练成功"""
    y = get_sample(type)
    y_before = torch.tensor(y, dtype=torch.float).to(DEVICE)
    y_after = predict(y_before)
    loss = loss_func(y_after, y_before)
    if show_plt:
        draw(y_before, y_after, type)
    return loss.item()


if __name__ == "__main__":
    train()
    test_cycle = 1
    results = {}
    for type in SUPPORTED_SAMPLE_TYPES:
        result = [test(type, show_plt=True) for _ in range(test_cycle)]
        results[type] = np.mean(result)
    print(results)
    plt.bar(range(len(results)), results.values(),
            tick_label=list(results.keys()))
    plt.title("abnormal weights")
    plt.show()
