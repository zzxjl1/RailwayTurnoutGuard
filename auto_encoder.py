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
from sensor import SAMPLE_RATE, SUPPORTED_SAMPLE_TYPES, generate_sample
from sensor.dataset import generate_dataset, get_sample

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


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


loss_func = nn.MSELoss()  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # 优化器


def get_dataloader(type):
    temp, _ = generate_dataset(dataset_length=TRAINING_SET_LENGTH + TESTING_SET_LENGTH,
                               time_series_length=TIME_SERIES_LENGTH,
                               type=type,
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


def train(type="normal"):
    train_dl, test_dl = get_dataloader(type)
    for epoch in range(EPOCHS):
        model.train()
        for i, (x,) in enumerate(train_dl):
            loss_batch(model, x, is_train=True)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, x, is_train=False) for (x,) in test_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print('Epoch [{}/{}], Validation_Loss: {}'
              .format(epoch + 1, EPOCHS, val_loss))

    torch.save(model, f"{FILE_PATH}{type}.pth")  # 保存模型


def train_all():
    for type in SUPPORTED_SAMPLE_TYPES:
        model.apply(init_weights)
        train(type)


def predict(x):
    assert len(
        x) == TIME_SERIES_LENGTH // POOLING_FACTOR_PER_TIME_SERIES * CHANNELS
    results = {}
    losses = {}
    for type in SUPPORTED_SAMPLE_TYPES:
        model_path = f"{FILE_PATH}{type}.pth"
        assert os.path.exists(
            model_path), f"model {type} not found, please train first"
        model = torch.load(model_path)
        model.eval()
        with torch.no_grad():
            result = model(x)
            results[type] = result
            loss = loss_func(result, x)
            losses[type] = loss.item()
    return results, losses


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


def test(type="normal", show_plt=False):
    """生成一个样本，并进行正向传播，如果输出与输入相似，则说明模型训练成功"""
    y, _ = get_sample(time_series_length=TIME_SERIES_LENGTH,
                      type=type,
                      pooling_factor_per_time_series=POOLING_FACTOR_PER_TIME_SERIES,
                      series_to_encode=SERIES_TO_ENCODE)
    y_before = torch.tensor(y, dtype=torch.float).to(DEVICE)
    y_before = y_before.view(TOTAL_LENGTH)
    results, losses = predict(y_before)
    for ae_type in SUPPORTED_SAMPLE_TYPES:
        loss = losses[ae_type]
        y_after = results[ae_type]
        if show_plt:
            draw(y_before, y_after,
                 f"Sample type: {type} - AutoEncoder type: {ae_type} - Loss: {loss}")
    if show_plt:
        plt.bar(range(len(losses)), losses.values(),
                tick_label=list(losses.keys()))
        plt.title(f"AutoEncoder Loss Result - Sample type: {type}")
        plt.show()

    return losses


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_result_matrix():
    d2_losses = []  # 二维loss矩阵
    for type in SUPPORTED_SAMPLE_TYPES:
        losses = test(type, show_plt=False)
        losses = list(losses.values())
        # 使用sigmoid函数将loss转换为概率
        losses = [sigmoid(loss) for loss in losses]
        # 翻转loss，使得loss越小，实际越大
        losses = [max(losses)-loss for loss in losses]
        # 放缩到0-1之间
        losses = [(loss - min(losses)) / (max(losses) - min(losses))
                  for loss in losses]
        d2_losses.append(losses)
    d2_losses = preprocessing.MinMaxScaler().fit_transform(d2_losses)  # 归一化
    return d2_losses


if __name__ == "__main__":

    # train_all()

    matrix = np.zeros((len(SUPPORTED_SAMPLE_TYPES),
                      len(SUPPORTED_SAMPLE_TYPES)))
    test_cycles = 5
    for i in range(test_cycles):
        matrix += np.array(get_result_matrix())
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
