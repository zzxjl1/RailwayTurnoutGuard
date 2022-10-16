"""
pytorch implementation of dnn classification
"""

import math
from re import X
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from extract_features import calc_features
from alive_progress import alive_bar, alive_it
from sensor import SUPPORTED_SAMPLE_TYPES, generate_sample


BATCH_SIZE = 32  # 每批处理的数据
FORCE_CPU = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() and not FORCE_CPU
                      else 'cpu')  # 放在cuda或者cpu上训练
print('Using device:', DEVICE)
EPOCHS = 10000  # 训练数据集的轮次
LEARNING_RATE = 1e-3  # 学习率


def weight_init(m):  # 初始化权重
    if isinstance(m, nn.Conv3d):
        n = m.kernel_size[0] * m.kernel_size[1] * \
            m.kernel_size[2] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1+8*3*4, 64)
        self.fc2 = nn.Linear(64, 128)
        self.out = nn.Linear(128, 12)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # softmax
        x = F.softmax(self.out(x), dim=1)
        return x


model = Net().to(DEVICE)
"""
model = nn.Sequential(
    nn.Linear(1+8*3*4, 64),
    nn.BatchNorm1d(64),
    nn.Sigmoid(),
    nn.Linear(64, 128),
    nn.BatchNorm1d(128),
    nn.Sigmoid(),
    nn.Linear(128, 12),
    nn.Softmax(dim=1)
).to(DEVICE)
"""

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_func = nn.CrossEntropyLoss()
model.apply(weight_init)  # 加载权重


def get_data(train_ds, valid_ds):
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True),
    )


def check(outputs):
    outputs = np.array(outputs)
    t = int((outputs != outputs).sum())
    if(t > 0):
        print("your data contains Nan")
        return False
    else:
        print("Your data does not contain Nan, it might be other problem")
        return True


def loss_batch(model, loss_func, xb, yb):
    optimizer.zero_grad()
    output = model(xb)
    """print("input", xb)
    print("output", output)
    print("target", yb)"""
    loss = loss_func(output, yb)
    loss.requires_grad_(True)
    loss.backward()
    optimizer.step()
    return loss.item(), len(xb)


def fit(train_dl, valid_dl):
    for step in range(EPOCHS):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print('当前step:' + str(step), '验证集损失：' + str(val_loss))


def generate_data(num):
    x = []
    y = []
    with alive_bar(num, title="数据集生成中") as bar:
        while len(x) < num:
            for sample_type in SUPPORTED_SAMPLE_TYPES:
                if len(x) >= num:
                    break
                sample = generate_sample(sample_type)
                t = list(calc_features(sample).values())
                # STANDARDIZE
                t = (t - np.mean(t)) / np.std(t)
                if not check(t):
                    break
                x.append(t)
                # one-hot encoding
                index = SUPPORTED_SAMPLE_TYPES.index(sample_type)
                y.append([0] * index + [1] + [0] *
                         (len(SUPPORTED_SAMPLE_TYPES) - index - 1))
                bar()
    return x, y


if __name__ == '__main__':
    DATASET_LENGTH = 100
    TRANING_SET_LENGTH = 80
    x, y = map(lambda a: torch.tensor(np.array(a), dtype=torch.float,
               requires_grad=True), generate_data(DATASET_LENGTH))
    x, y = x.to(DEVICE), y.to(DEVICE)
    print(x.shape, y.shape)
    #print(x[0], y[0])
    train_ds = TensorDataset(x[:TRANING_SET_LENGTH], y[:TRANING_SET_LENGTH])
    valid_ds = TensorDataset(
        x[TRANING_SET_LENGTH:], y[TRANING_SET_LENGTH:])
    train_dl, valid_dl = get_data(train_ds, valid_ds)
    fit(train_dl, valid_dl)


"""
TODO: 修复生成数据时的会有nan（见check那行）
"""
