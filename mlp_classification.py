"""
pytorch implementation of dnn classification
用dnn对提取的特征进行分类
至此论文复现完毕
"""

import os
import pickle
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from extract_features import IGNORE_LIST, calc_features, SERIES_TO_ENCODE
from alive_progress import alive_bar
from sensor import SUPPORTED_SAMPLE_TYPES, get_sample
from gru_score import GRUScore
from sensor.real_world import get_all_samples
from tool_utils import get_label_from_result_pretty, parse_predict_result

FILE_PATH = "./models/mlp_classification.pth"
BATCH_SIZE = 64  # 每批处理的数据
FORCE_CPU = False  # 强制使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
print("Using device:", DEVICE)
EPOCHS = 200  # 训练数据集的轮次
LEARNING_RATE = 1e-3  # 学习率

INPUT_VECTOR_SIZE = 1 + 15 * len(SERIES_TO_ENCODE) * 3 - len(IGNORE_LIST)  # 输入向量的大小
TRANING_SET_LENGTH = 400  # 训练集长度
TESTING_SET_LENGTH = 100  # 测试集长度
DATASET_LENGTH = TRANING_SET_LENGTH + TESTING_SET_LENGTH
N_CLASSES = len(SUPPORTED_SAMPLE_TYPES)  # 分类数


def weight_init(m):
    """初始化权重"""
    if isinstance(m, nn.Conv3d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()


class MLP(nn.Module):
    def __init__(self, input_vector_size, output_vector_size):
        super(MLP, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_vector_size)
        self.fc1 = nn.Linear(input_vector_size, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, output_vector_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        bn1_result = self.bn1(x)

        fc1_result = self.fc1(bn1_result)
        bn2_result = self.bn2(fc1_result)
        x = F.relu(bn2_result)

        fc2_result = self.fc2(x)
        bn3_result = self.bn3(fc2_result)
        x = F.relu(bn3_result)

        out = self.out(x)
        softmax_result = self.softmax(out)
        return softmax_result


"""
# 定义BP模型结构
BP_Net = nn.Sequential(
    nn.BatchNorm1d(INPUT_VECTOR_SIZE),  # 归一化
    nn.Linear(INPUT_VECTOR_SIZE, 64),  # 全连接层
    nn.BatchNorm1d(64),
    nn.ReLU(),  # 激活函数
    nn.Linear(64, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128, 12),
    nn.Softmax(dim=1)  # 分类任务最后用softmax层
)
"""


model = MLP(input_vector_size=INPUT_VECTOR_SIZE, output_vector_size=N_CLASSES).to(
    DEVICE
)  # 使用BP模型
print(model)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # adam优化器
loss_func = nn.CrossEntropyLoss()  # 交叉熵损失函数
model.apply(weight_init)  # 预初始化权重


def get_data(train_ds, valid_ds):  # 获取dataloader
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True),
    )


def contains_nan(outputs):
    """判断是否有nan,nan会导致训练失败"""
    outputs = np.array(outputs)
    t = int((outputs != outputs).sum())
    return t > 0


def loss_batch(model, loss_func, xb, yb):
    optimizer.zero_grad()
    output = model(xb)  # 前向传播
    """
    print("input", xb)
    print("output", output)
    print("target", yb)
    """
    loss = loss_func(output, yb)  # 计算损失
    loss.requires_grad_(True)
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    return loss.item(), len(xb)


def fit(train_dl, valid_dl):
    for step in range(EPOCHS):  # 训练轮次
        model.train()  # 训练模式
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb)

        model.eval()  # 验证模式
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print("当前step:" + str(step), "验证集损失：" + str(val_loss))


def generate_dataset():
    """生成数据集"""
    x = []  # 特征值
    y = []  # 目标值

    samples, types = get_all_samples()  # 获取所有样本
    assert len(samples) >= DATASET_LENGTH
    samples, types = samples[:DATASET_LENGTH], types[:DATASET_LENGTH]  # 取前num个样本

    with alive_bar(DATASET_LENGTH, title="数据集生成中") as bar:
        for sample, sample_type in zip(samples, types):
            t = list(calc_features(sample).values())  # 计算特征
            # STANDARDIZE
            # t = (t - np.mean(t)) / np.std(t)
            assert not contains_nan(t)  # 检查是否有nan
            x.append(t)
            # one-hot encoding
            index = SUPPORTED_SAMPLE_TYPES.index(sample_type)
            y.append(
                [0] * index + [1] + [0] * (len(SUPPORTED_SAMPLE_TYPES) - index - 1)
            )
            bar()  # 进度条+1

    x = torch.tensor(np.array(x), dtype=torch.float, requires_grad=True)
    y = torch.tensor(np.array(y), dtype=torch.float, requires_grad=True)
    x, y = x.to(DEVICE), y.to(DEVICE)
    print(x.shape, y.shape)
    # print(x[0], y[0])
    train_ds = TensorDataset(x[:TRANING_SET_LENGTH], y[:TRANING_SET_LENGTH])  # 训练集
    valid_ds = TensorDataset(x[TRANING_SET_LENGTH:], y[TRANING_SET_LENGTH:])  # 验证集
    return train_ds, valid_ds


def predict_raw_input(x):
    """预测,输入为原始数据，直接入模型"""
    assert os.path.exists(FILE_PATH), "model not found, please run train() first!"
    model = torch.load(FILE_PATH, map_location=DEVICE).to(DEVICE)  # 加载模型
    model.eval()  # 验证模式
    with torch.no_grad():
        output = model(x)
        return output


def train():
    """训练模型"""
    fit(train_dl, valid_dl)  # 开始训练

    torch.save(model, FILE_PATH)  # 保存模型
    # torch.onnx.export(model, torch.randn(1, INPUT_VECTOR_SIZE),"model.onnx")  # 保存onnx格式模型


def predict(sample, segmentations=None):
    """预测"""
    features = list(calc_features(sample, segmentations).values())  # 计算特征
    features = torch.tensor([features], dtype=torch.float)  # 转换为tensor
    result = predict_raw_input(features.to(DEVICE))  # 预测
    return result.squeeze()


def test():
    model = torch.load(FILE_PATH, map_location=DEVICE).to(DEVICE)  # 加载模型
    model.eval()  # 验证模式
    correct = 0
    total = 0
    for i, (x, y) in enumerate(valid_dl):
        y = y.float().to(DEVICE)
        output = model(x)
        _, predicted = torch.max(output.data, 1)
        _, label = torch.max(y.data, 1)
        total += y.size(0)
        correct += (predicted == label).sum().item()
    print("accu:", correct / total)


if __name__ == "__main__":
    train_ds, valid_ds = generate_dataset()  # 生成数据集
    train_dl, valid_dl = get_data(train_ds, valid_ds)  # 转换为dataloader
    # train()  # 训练模型，第一次运行时需要先训练模型，训练完会持久化权重至硬盘请注释掉这行
    test()  # 测试模型
