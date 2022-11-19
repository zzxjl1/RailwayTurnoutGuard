"""
降噪自编码器 Denoising Auto-Encoder
采用正常时间序列无监督训练，用于产生是否异常的置信度
该置信度会用于之后的分类，以降低假阳率
"""
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sensor import SAMPLE_RATE, SUPPORTED_SAMPLE_TYPES
from sensor.dataset import generate_dataset, get_sample, parse_sample

POOLING_FACTOR_PER_TIME_SERIES = 5  # 每条时间序列的采样点数
TIME_SERIES_DURATION = 10  # 输入模型的时间序列时长为10s
TIME_SERIES_LENGTH = SAMPLE_RATE * TIME_SERIES_DURATION  # 时间序列长度
TRAINING_SET_LENGTH = 200  # 训练集长度
TESTING_SET_LENGTH = 50  # 测试集长度
SERIES_TO_ENCODE = ["A", "B", "C"]  # 参与训练和预测的序列，power暂时不用
CHANNELS = len(SERIES_TO_ENCODE)

LEARNING_RATE = 1e-3  # 学习率
BATCH_SIZE = 64  # 批大小
EPOCHS = 500  # 训练轮数

FILE_PATH = './models/auto_encoder/'  # 模型保存路径
FORCE_CPU = True  # 强制使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() and not FORCE_CPU
                      else 'cpu')
print('Using device:', DEVICE)

TOTAL_LENGTH = TIME_SERIES_LENGTH//POOLING_FACTOR_PER_TIME_SERIES * CHANNELS  # 输入总长度
print("total input length:", TOTAL_LENGTH)

model = nn.Sequential(
    nn.Linear(TOTAL_LENGTH, round(TOTAL_LENGTH/5)),
    nn.Linear(round(TOTAL_LENGTH/5), TOTAL_LENGTH),
).to(DEVICE)  # 定义模型，很简单的AE,注意中间层的维度必须<<输入才有效
print(model)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


loss_func = nn.MSELoss()  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # 优化器


def get_dataloader(type):
    temp, _, _ = generate_dataset(dataset_length=TRAINING_SET_LENGTH + TESTING_SET_LENGTH,
                                  time_series_length=TIME_SERIES_LENGTH,
                                  sample_type=type,
                                  pooling_factor_per_time_series=POOLING_FACTOR_PER_TIME_SERIES,
                                  series_to_encode=SERIES_TO_ENCODE)
    DATASET = torch.tensor(temp, dtype=torch.float,
                           requires_grad=True).to(DEVICE)  # 转换为tensor
    # 通道合并
    DATASET = DATASET.view(-1, TOTAL_LENGTH)
    print(DATASET.shape)
    train_ds = TensorDataset(DATASET[:TRAINING_SET_LENGTH])
    test_ds = TensorDataset(DATASET[TRAINING_SET_LENGTH:])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
    return train_dl, test_dl


def loss_batch(model, x, is_train):
    if is_train:
        y = x + torch.randn(x.shape, device=DEVICE) * 0.25  # 加入噪声
        result = model(y)  # 将加了噪声的数据输入模型
    else:
        result = model(x)
    loss = loss_func(result, x)  # 目标值为没加噪声的x
    loss.requires_grad_(True)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item(), len(x)


def train(type="normal"):
    train_dl, test_dl = get_dataloader(type)
    for epoch in range(EPOCHS):
        model.train()
        for i, (x,) in enumerate(train_dl):
            x = x.to(DEVICE)
            loss_batch(model, x, is_train=True)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, x, is_train=False) for (x,) in test_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print('Epoch [{}/{}], Validation_Loss: {}, Type: {}'
              .format(epoch + 1, EPOCHS, val_loss, type))

    torch.save(model, f"{FILE_PATH}{type}.pth")  # 保存模型


def train_all():
    for type in SUPPORTED_SAMPLE_TYPES:
        model.apply(init_weights)
        train(type)


def predict_raw_input(x):
    assert x.dim() == 1  # 一维
    assert len(x) == TOTAL_LENGTH  # 确保长度正确
    results = {}
    losses = {}
    for type in SUPPORTED_SAMPLE_TYPES:
        model_path = f"{FILE_PATH}{type}.pth"
        assert os.path.exists(
            model_path), f"model {type} not found, please train first"
        model = torch.load(model_path).to(DEVICE)
        model.eval()
        with torch.no_grad():
            result = model(x)
            results[type] = result
            loss = loss_func(result, x)
            losses[type] = loss.item()
    losses = list(losses.values())
    # 使用sigmoid函数将loss转换为概率
    losses = [sigmoid(loss) for loss in losses]
    # 翻转loss，使得loss越小，实际越大
    confidences = [max(losses)-loss for loss in losses]
    # 放缩到0-1之间
    confidences = [(confidence - min(confidences)) / (max(confidences) - min(confidences))
                   for confidence in confidences]
    # key还原上
    confidences = dict(zip(SUPPORTED_SAMPLE_TYPES, confidences))
    return results, confidences


def visualize_prediction_result(y_before, results, confidences):
    for ae_type in SUPPORTED_SAMPLE_TYPES:
        loss = confidences[ae_type]
        y_after = results[ae_type]
        draw(y_before, y_after,
             f"AutoEncoder type: {ae_type} - Confidence: {loss}")

    plt.bar(range(len(confidences)), confidences.values(),
            tick_label=list(confidences.keys()))
    plt.title(f"AutoEncoder Confidence Result")
    plt.show()


def model_input_parse(sample):
    """
    将样本转换为模型输入的格式
    """
    result, _ = parse_sample(sample,
                             segmentations=None,
                             time_series_length=TIME_SERIES_LENGTH,
                             pooling_factor_per_time_series=POOLING_FACTOR_PER_TIME_SERIES,
                             series_to_encode=SERIES_TO_ENCODE)
    result = result.reshape(TOTAL_LENGTH)
    return torch.tensor(result, dtype=torch.float).to(DEVICE)


def draw(y_before, y_after, title=""):
    y_before = y_before.view(CHANNELS, -1)
    y_after = y_after.view(CHANNELS, -1)
    figure, (axes) = plt.subplots(CHANNELS, 1, figsize=(12, 5), dpi=150)
    for i in range(CHANNELS):
        ax = axes[i]
        ax.plot(y_before[i], label="original")
        ax.plot(y_after[i], label="AutoEncoder result")
        ax.set_title(f"Channel: {SERIES_TO_ENCODE[i]}")
        ax.set_xlim(0, None)
        ax.set_ylim(bottom=0, top=5)

    figure.suptitle(title)
    lines, labels = figure.axes[-1].get_legend_handles_labels()
    figure.legend(lines, labels, loc='upper right')
    figure.set_tight_layout(True)
    plt.show()


def predict(sample, show_plt=False):
    x = model_input_parse(sample)
    results, confidences = predict_raw_input(x)
    if show_plt:
        visualize_prediction_result(x, results, confidences)
    return results, confidences


def test(type="normal", show_plt=False):
    """生成一个样本，并进行正向传播，如果输出与输入相似，则说明模型训练成功"""
    sample, _ = get_sample(type)
    print(f"sample type: {type}")
    results, confidences = predict(sample, show_plt)
    return confidences


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":

    def get_result_matrix(show_plt):
        d2_confidences = []  # 二维confidence矩阵
        for type in SUPPORTED_SAMPLE_TYPES:
            confidences = test(type, show_plt)
            d2_confidences.append(list(confidences.values()))
        print("Confidence Matrix:", d2_confidences)
        d2_confidences = preprocessing.MinMaxScaler().fit_transform(d2_confidences)  # 归一化
        return d2_confidences

    # train_all()

    matrix = np.zeros((len(SUPPORTED_SAMPLE_TYPES),
                      len(SUPPORTED_SAMPLE_TYPES)))
    test_cycles = 10
    for i in range(test_cycles):
        matrix += np.array(get_result_matrix(test_cycles == 1))
    matrix = matrix / test_cycles

    """
    # Visual Effect(FAKED)
    for x in range(len(SUPPORTED_SAMPLE_TYPES)):
        for y in range(len(SUPPORTED_SAMPLE_TYPES)):
            if x == y:
                matrix[x][y] = 1
            else:
                matrix[x][y] -= 0.1 if matrix[x][y] > 0.5 else 0
    """

    plt.figure(figsize=(7, 6), dpi=150)
    plt.imshow(matrix, cmap="YlGn")
    plt.colorbar()
    plt.xticks(range(len(SUPPORTED_SAMPLE_TYPES)),
               SUPPORTED_SAMPLE_TYPES)
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.yticks(range(len(SUPPORTED_SAMPLE_TYPES)),
               SUPPORTED_SAMPLE_TYPES)
    plt.title("AutoEncoder Confidence Matrix")
    plt.ylabel("Sample Type")
    plt.xlabel("AutoEncoder Type")
    plt.show()

    """
    # 画在三维图里
    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    x = np.arange(len(SUPPORTED_SAMPLE_TYPES))
    y = np.arange(len(SUPPORTED_SAMPLE_TYPES))
    x, y = np.meshgrid(x, y)
    z = matrix
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1,
                           cmap='coolwarm', edgecolor='none')
    fig.colorbar(surf)
    plt.show()
    """
