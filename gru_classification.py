import os
import random
from alive_progress import alive_bar
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from segmentation import calc_segmentation_points
from sensor import SUPPORTED_SAMPLE_TYPES
from sensor import generate_dataset, parse_sample, get_sample
from sensor.config import SAMPLE_RATE
from gru_score import GRUScore
from tool_utils import (
    get_label_from_result_pretty,
    parse_predict_result,
    show_confusion_matrix,
)

from torch.utils.tensorboard import SummaryWriter
from pytorchtools import EarlyStopping  # Add an EarlyStopping utility

writer = SummaryWriter("./paper/tcn")  # TensorBoard writer
early_stopping = EarlyStopping(patience=20, verbose=True)


FILE_PATH = "./models/gru_classification.pth"
TRAINING_SET_LENGTH = 400  # 训练集长度
TESTING_SET_LENGTH = 100  # 测试集长度
DATASET_LENGTH = TRAINING_SET_LENGTH + TESTING_SET_LENGTH  # 数据集总长度
LEARNING_RATE = 1e-4  # 学习率
BATCH_SIZE = 128  # 每批处理的数据
FORCE_CPU = False  # 强制使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
N_CLASSES = len(SUPPORTED_SAMPLE_TYPES)  # 分类数
SERIES_TO_ENCODE = ["A", "B", "C"]
CHANNELS = len(SERIES_TO_ENCODE)
TIME_SERIES_DURATION = 15  # 15s
TIME_SERIES_LENGTH = SAMPLE_RATE * TIME_SERIES_DURATION  # 采样率*时间，总共的数据点数
POOLING_FACTOR_PER_TIME_SERIES = 3  # 每个时间序列的池化因子,用于降低工作量
SEQ_LENGTH = TIME_SERIES_LENGTH // POOLING_FACTOR_PER_TIME_SERIES  # 降采样后的序列长度


class Vanilla_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Vanilla_GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, seq):
        hidden = torch.zeros(self.num_layers, seq.size(0), self.hidden_size).to(DEVICE)
        # adj_seq = seq.permute(self.batch_size, len(seq), -1)
        output, hidden = self.gru(seq, hidden)
        return output, hidden


class Squeeze_Excite(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, s = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excite(y).view(b, c, 1)
        return x * y.expand_as(x)


class FCN_1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCN_1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=8,
            padding=4,
            padding_mode="replicate",
        )
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm1d(out_channels, eps=1e-03, momentum=0.99)
        self.SE1 = Squeeze_Excite(out_channels)

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels * 2,
            kernel_size=5,
            padding=2,
            padding_mode="replicate",
        )
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm1d(out_channels * 2, eps=1e-03, momentum=0.99)
        self.SE2 = Squeeze_Excite(out_channels * 2)

        self.conv3 = nn.Conv1d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.bn3 = nn.BatchNorm1d(out_channels, eps=1e-03, momentum=0.99)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, seq):
        adj_seq = seq.permute(0, 2, 1)

        y = self.conv1(adj_seq)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.SE1(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.SE2(y)

        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu(y)

        y = self.gap(y)

        return y


class GRU_FCN(nn.Module):
    def __init__(self, seq_len, n_class, dropout_rate, hidden_size):
        super().__init__()
        self.GRU_model = Vanilla_GRU(
            input_size=CHANNELS, hidden_size=hidden_size, num_layers=1
        ).to(DEVICE)
        self.FCN_model = FCN_1D(in_channels=CHANNELS, out_channels=hidden_size).to(
            DEVICE
        )
        self.seq_len = seq_len

        self.dropout = nn.Dropout(p=dropout_rate)
        self.Dense = nn.Linear(in_features=hidden_size * 2, out_features=n_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, seq):
        y_GRU, _ = self.GRU_model(seq)
        y_GRU = y_GRU.transpose(0, 1)[-1]
        y_GRU = self.dropout(y_GRU)
        y_FCN = self.FCN_model(seq).squeeze()
        if len(y_FCN.size()) == 1:
            y_FCN = y_FCN.unsqueeze(0)
        concat = torch.cat([y_GRU, y_FCN], 1)
        y = self.Dense(concat)
        y = self.softmax(y)
        return y


model = GRU_FCN(
    seq_len=SEQ_LENGTH, n_class=N_CLASSES, dropout_rate=0.2, hidden_size=128
).to(DEVICE)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def model_input_parse(sample, segmentations=None, batch_simulation=True):
    """
    将样本转换为模型输入的格式
    """
    if segmentations is None:
        segmentations = calc_segmentation_points(sample)
    sample_array, seg_index = parse_sample(
        sample,
        segmentations,
        time_series_length=TIME_SERIES_LENGTH,
        pooling_factor_per_time_series=POOLING_FACTOR_PER_TIME_SERIES,
        series_to_encode=SERIES_TO_ENCODE,
    )
    x = sample_array.transpose()

    if batch_simulation:
        x = x[np.newaxis, :, :]

    return x


def get_dataloader():
    X, _, labels = generate_dataset(
        dataset_length=TRAINING_SET_LENGTH + TESTING_SET_LENGTH,
        time_series_length=TIME_SERIES_LENGTH,
        sample_type=None,
        pooling_factor_per_time_series=POOLING_FACTOR_PER_TIME_SERIES,
        series_to_encode=SERIES_TO_ENCODE,
        no_segmentation=True,
    )
    X = torch.tensor(X, dtype=torch.float, requires_grad=True).to(DEVICE)  # 转换为tensor
    X = X.transpose(1, 2)  # 转换为(batch_size, channels, seq_len)的格式
    Y = []
    for label in labels:
        index = SUPPORTED_SAMPLE_TYPES.index(label)
        Y.append([0] * index + [1] + [0] * (len(SUPPORTED_SAMPLE_TYPES) - index - 1))
    Y = torch.tensor(Y, dtype=torch.float, requires_grad=True).to(DEVICE)

    train_ds = TensorDataset(X[:TRAINING_SET_LENGTH], Y[:TRAINING_SET_LENGTH])
    test_ds = TensorDataset(X[TRAINING_SET_LENGTH:], Y[TRAINING_SET_LENGTH:])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
    return train_dl, test_dl


def train():
    model.train()
    epoch = 0
    train_loss = []
    while 1:  # 训练EPOCHS轮
        for i, (x, y) in enumerate(train_dl):
            y = y.float().to(DEVICE)
            optimizer.zero_grad()  # 梯度清零
            output = model(x)  # 前向传播
            l = loss(output, y)  # 计算损失
            train_loss.append(l.item())
            l.backward()  # 反向传播
            optimizer.step()  # 更新参数
            print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, i, l.item()))
        # torch.save(model, FILE_PATH)  # 保存模型
        val_loss = test()
        writer.add_scalar("Loss/Train", sum(train_loss) / len(train_loss), epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        epoch += 1

        # Add early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def predict_raw_input(x):
    assert os.path.exists(FILE_PATH), "model not found，please train first"
    model = torch.load(FILE_PATH, map_location=DEVICE).to(DEVICE)  # 加载模型

    # 转tensor
    x = torch.tensor(x, dtype=torch.float32).to(DEVICE)

    model.eval()  # 验证模式
    with torch.no_grad():
        output = model(x)
    return output


def predict(sample, segmentations=None):
    x = model_input_parse(sample, segmentations, batch_simulation=True)  # 转换为模型输入格式
    output = predict_raw_input(x)
    return output.squeeze()


def test():
    # model = torch.load(FILE_PATH, map_location=DEVICE).to(DEVICE)  # 加载模型
    model.eval()  # 验证模式
    y_true = []
    y_pred = []
    losses = []
    for i, (x, y) in enumerate(test_dl):
        y = y.float().to(DEVICE)
        with torch.no_grad():
            output = model(x)
            losses.append(loss(output, y))
        _, predicted = torch.max(output.data, 1)
        _, label = torch.max(y.data, 1)
        y_true.extend(label.tolist())
        y_pred.extend(predicted.tolist())
    # report = classification_report(y_true, y_pred, target_names=SUPPORTED_SAMPLE_TYPES)
    # cm = confusion_matrix(y_true, y_pred)
    # print(report)
    print("eval loss:", sum(losses) / len(losses))
    # show_confusion_matrix(cm, SUPPORTED_SAMPLE_TYPES)
    return sum(losses) / len(losses)


if __name__ == "__main__":
    train_dl, test_dl = get_dataloader()
    train()

    # test()
