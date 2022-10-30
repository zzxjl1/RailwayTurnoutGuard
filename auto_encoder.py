"""
降噪自编码器 Denoising Auto-Encoder
采用正常时间序列无监督训练，用于产生是否异常的置信度
该置信度会用于之后的分类，以降低假阳率
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sensor import SAMPLE_RATE, SUPPORTED_SAMPLE_TYPES, generate_sample

TIME_SERIES_LENGTH = SAMPLE_RATE * 20  # 20s的时间序列采样点数
TRAINING_SET_LENGTH = 200
TESTING_SET_LENGTH = 50

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 500

FILENAME = './models/auto_encoder.pth'
FORCE_CPU = True  # 强制使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() and not FORCE_CPU
                      else 'cpu')
print('Using device:', DEVICE)


model = nn.Sequential(
    nn.Linear(TIME_SERIES_LENGTH, round(TIME_SERIES_LENGTH/5)),
    nn.Linear(round(TIME_SERIES_LENGTH/5), TIME_SERIES_LENGTH),
).to(DEVICE)

loss_func = nn.MSELoss()  # 均方误差
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # 优化器


def parse_input(phase_a_time_series):
    # 超长的截断，短的补0
    if len(phase_a_time_series) > TIME_SERIES_LENGTH:
        return np.array(
            phase_a_time_series[:TIME_SERIES_LENGTH])
    else:
        return np.pad(
            phase_a_time_series, (0, TIME_SERIES_LENGTH - len(phase_a_time_series)), 'constant')


def generate_data():
    DATASET_LENGTH = TRAINING_SET_LENGTH + TESTING_SET_LENGTH
    DATASET = []
    for _ in range(DATASET_LENGTH):
        temp, _ = generate_sample(type="normal")
        phase_a_time_series = parse_input(temp["A"][1])
        DATASET.append(phase_a_time_series)
    return DATASET


def get_dataloader():
    temp = generate_data()
    DATASET = torch.tensor(np.array(temp), dtype=torch.float,
                           requires_grad=True).to(DEVICE)
    print(DATASET.shape)
    train_ds = TensorDataset(DATASET[:TRAINING_SET_LENGTH])
    test_ds = TensorDataset(DATASET[TRAINING_SET_LENGTH:])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
    return train_dl, test_dl


def loss_batch(model, x):
    x += torch.randn(x.shape) * 0.5

    result = model(x)
    loss = loss_func(result, x)
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
            loss_batch(model, x[0])

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, x[0]) for x in test_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)

    torch.save(model, FILENAME)


def predict(x):
    if not os.path.exists(FILENAME):
        train()
    model = torch.load(FILENAME)
    model.eval()
    with torch.no_grad():
        return model(x)


def draw(y_before, y_after, title=""):
    import matplotlib.pyplot as plt
    plt.plot(y_before)
    plt.plot(y_after)
    plt.title(title)
    plt.show()


def test(type="normal", show_plt=False):
    """生成一个正常样本，并进行正向传播，如果输出与输入相似，则说明模型训练成功"""
    y, _ = generate_sample(type)
    y_before = torch.tensor(parse_input(
        y["A"][1]), dtype=torch.float).to(DEVICE)

    y_after = predict(y_before)
    loss = loss_func(y_after, y_before)
    if show_plt:
        draw(y_before, y_after, type)
    return loss.item()


if __name__ == "__main__":
    train()
    test_cycle = 10
    show_plt = False
    results = {}
    for type in SUPPORTED_SAMPLE_TYPES:
        result = [test(type, show_plt) for _ in range(test_cycle)]
        results[type] = np.mean(result)
    print(results)
