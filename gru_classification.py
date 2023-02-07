import os
import random
from alive_progress import alive_bar
from matplotlib import patches, pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from segmentation import calc_segmentation_points
from sensor import SUPPORTED_SAMPLE_TYPES
from sensor.dataset import get_sample, generate_dataset, parse_sample
from sensor.config import SAMPLE_RATE
from gru_score import GRUScore
from tool_utils import get_label_from_result_pretty, parse_predict_result

FILE_PATH = "./models/gru_classification.pth"
DATASET_LENGTH = 500  # 数据集总长度
EPOCHS = 100  # 训练数据集的轮次
LEARNING_RATE = 1e-4  # 学习率
BATCH_SIZE = 64  # 每批处理的数据
FORCE_CPU = True  # 强制使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available()
                      and not FORCE_CPU else 'cpu')
N_CLASSES = len(SUPPORTED_SAMPLE_TYPES)  # 分类数
SERIES_TO_ENCODE = ["A", "B", "C"]
CHANNELS = len(SERIES_TO_ENCODE)
TIME_SERIES_DURATION = 20  # 20s
TIME_SERIES_LENGTH = SAMPLE_RATE * TIME_SERIES_DURATION  # 采样率*时间，总共的数据点数
POOLING_FACTOR_PER_TIME_SERIES = 5  # 每个时间序列的池化因子,用于降低工作量
SEQ_LENGTH = TIME_SERIES_LENGTH // POOLING_FACTOR_PER_TIME_SERIES  # 降采样后的序列长度


class Vanilla_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Vanilla_GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size,
                          hidden_size,
                          batch_first=True)

    def forward(self, seq):
        hidden = torch.zeros(self.num_layers,
                             seq.size(0),
                             self.hidden_size).to(DEVICE)
        # adj_seq = seq.permute(self.batch_size, len(seq), -1)
        output, hidden = self.gru(seq, hidden)
        return output, hidden


class Squeeze_Excite(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(channel // reduction,
                                              channel, bias=False),
                                    nn.Sigmoid()
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
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=8,
                               padding=4,
                               padding_mode='replicate'
                               )
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm1d(128, eps=1e-03, momentum=0.99)
        self.SE1 = Squeeze_Excite(128)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels*2,
                               kernel_size=5,
                               padding=2,
                               padding_mode='replicate'
                               )
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm1d(256, eps=1e-03, momentum=0.99)
        self.SE2 = Squeeze_Excite(256)

        self.conv3 = nn.Conv1d(in_channels=out_channels*2,
                               out_channels=out_channels,
                               kernel_size=3,
                               padding=1,
                               padding_mode='replicate'
                               )
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.bn3 = nn.BatchNorm1d(128, eps=1e-03, momentum=0.99)
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
    def __init__(self, GRU, FCN,  seq_len, n_class, dropout_rate, concat_features):
        super().__init__()
        self.GRU = GRU
        self.FCN = FCN
        self.seq_len = seq_len

        self.dropout = nn.Dropout(p=dropout_rate)
        self.Dense = nn.Linear(
            in_features=concat_features, out_features=n_class)

    def forward(self, seq):
        y_GRU, _ = self.GRU(seq)
        y_GRU = y_GRU.transpose(0, 1)[-1]
        y_GRU = self.dropout(y_GRU)
        y_FCN = self.FCN(seq).squeeze()
        if len(y_FCN.size()) == 1:
            y_FCN = y_FCN.unsqueeze(0)
        concat = torch.cat([y_GRU, y_FCN], 1)
        y = self.Dense(concat)
        return y


GRU_model = Vanilla_GRU(input_size=CHANNELS,
                        hidden_size=128,
                        num_layers=1).to(DEVICE)
FCN_model = FCN_1D(in_channels=CHANNELS, out_channels=128).to(DEVICE)
model = GRU_FCN(GRU=GRU_model,
                FCN=FCN_model,
                seq_len=SEQ_LENGTH,
                n_class=N_CLASSES,
                dropout_rate=0.2,
                concat_features=128+128).to(DEVICE)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def model_input_parse(sample, segmentations=None, batch_simulation=True):
    """
    将样本转换为模型输入的格式
    """
    if segmentations is None:
        segmentations = calc_segmentation_points(sample)
    sample_array, seg_index = parse_sample(sample,
                                           segmentations,
                                           time_series_length=TIME_SERIES_LENGTH,
                                           pooling_factor_per_time_series=POOLING_FACTOR_PER_TIME_SERIES,
                                           series_to_encode=SERIES_TO_ENCODE)
    x = sample_array.transpose()

    if batch_simulation:
        x = x[np.newaxis, :, :]

    return x


class Dataset(Dataset):
    def __init__(self, dataset_length):
        self.y = []  # 目标值
        self.x = []  # 特征值
        self.generate_dataset(dataset_length)

    def generate_dataset(self, num):
        with alive_bar(num, title="数据集生成中") as bar:
            while len(self.x) < num:  # 生成num个数据
                assert len(self.x) == len(self.y)
                sample_type = random.choice(SUPPORTED_SAMPLE_TYPES)
                sample, segmentations = get_sample(sample_type)  # 生成样本
                x = model_input_parse(
                    sample,
                    segmentations,
                    batch_simulation=False
                )  # 转换为模型输入格式
                self.x.append(x.astype(np.float32))
                # one-hot encoding
                index = SUPPORTED_SAMPLE_TYPES.index(sample_type)
                self.y.append([0] * index + [1] + [0] *
                              (len(SUPPORTED_SAMPLE_TYPES) - index - 1))
                bar()  # 进度条+1

    def __len__(self):
        assert len(self.x) == len(self.y)
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, np.array(y)


def train():
    dataset = Dataset(DATASET_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True)
    for epoch in range(EPOCHS):  # 训练EPOCHS轮
        for i, (x, y) in enumerate(loader):
            y = y.float().to(DEVICE)
            optimizer.zero_grad()  # 梯度清零
            output = model(x)  # 前向传播
            l = loss(output, y)  # 计算损失
            l.backward()  # 反向传播
            optimizer.step()  # 更新参数
            print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, i, l.item()))
    torch.save(model, FILE_PATH)  # 保存模型


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
    x = model_input_parse(
        sample,
        segmentations,
        batch_simulation=True
    )  # 转换为模型输入格式
    output = predict_raw_input(x)
    return output.squeeze()


def test(type=None, show_plt=False):
    if type is None:
        type = random.choice(SUPPORTED_SAMPLE_TYPES)
    sample, segmentations = get_sample(type)  # 生成样本
    output = predict(sample, segmentations, show_plt)
    result_pretty = parse_predict_result(output)  # 解析结果
    print(result_pretty)
    label = get_label_from_result_pretty(result_pretty)  # 获取结果
    print(type, label)
    return label == type


if __name__ == "__main__":
    train()

    # test(type="H5")

    test_cycle = 200
    accuracy = sum([test() for _ in range(test_cycle)])/test_cycle
    print("accuracy: {}".format(accuracy))
