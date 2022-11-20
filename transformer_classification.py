"""
依据分段结果，使用Transformer逐段提取特征后进行分类
这样避免漏掉手动特征工程无法涵盖的特征
之后会和dnn、ae的结果进行融合
"""
import os
import random
from alive_progress import alive_bar
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

FILE_PATH = "./models/transformer_classification.pth"
DATASET_LENGTH = 500  # 数据集总长度
EPOCHS = 10000  # 训练数据集的轮次
LEARNING_RATE = 1e-4  # 学习率
BATCH_SIZE = 64  # 每批处理的数据
FORCE_CPU = True  # 强制使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available()
                      and not FORCE_CPU else 'cpu')
N_CLASSES = len(SUPPORTED_SAMPLE_TYPES)  # 分类数
SERIES_TO_ENCODE = ["A", "B", "C", "power"]
CHANNELS = len(SERIES_TO_ENCODE)
TIME_SERIES_DURATION = 20  # 20s
TIME_SERIES_LENGTH = SAMPLE_RATE * TIME_SERIES_DURATION  # 采样率*时间，总共的数据点数
POOLING_FACTOR_PER_TIME_SERIES = 5  # 每个时间序列的池化因子,用于降低工作量
SEQ_LENGTH = TIME_SERIES_LENGTH // POOLING_FACTOR_PER_TIME_SERIES  # 降采样后的序列长度

STAGE_1_DURATION = 2
STAGE_2_DURATION = 20
STAGE_3_DURATION = 10


def time_to_index(sec):
    return int(sec * SAMPLE_RATE // POOLING_FACTOR_PER_TIME_SERIES)


class TransformerLayer(nn.Module):
    def __init__(self, input_vector_size, dropout_rate=0.1):
        super(TransformerLayer, self).__init__()
        self.input_vector_size = input_vector_size
        self.transformer = nn.TransformerEncoderLayer(input_vector_size,
                                                      nhead=1,
                                                      dim_feedforward=512,
                                                      dropout=dropout_rate,
                                                      batch_first=True,
                                                      device=DEVICE)
        #self.fc = nn.Linear(input_vector_size, CHANNELS)
        self.out = nn.Linear(input_vector_size, 1)

    def forward(self, x):
        x = self.transformer(x)
        #x = self.fc(x)
        x = self.out(x)
        x = x.squeeze(2)
        return x


"""
class RNNLayer(nn.Module):
    def __init__(self, input_vector_size, hidden_size=128, num_layers=1, dropout_rate=0.5):
        super(RNNLayer, self).__init__()
        self.input_vector_size = input_vector_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_vector_size,
                          hidden_size,
                          num_layers=num_layers,
                          dropout=dropout_rate if num_layers > 1 else 0,
                          batch_first=True)

        self.fc = nn.Linear(hidden_size, hidden_size//2)
        self.out = nn.Linear(hidden_size//2, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers,
                         x.size(0),
                         self.hidden_size).to(DEVICE)
        # c0 = torch.zeros(self.num_layers,
        #                 x.size(0),
        #                 self.hidden_size).to(DEVICE)
        #x, _ = self.rnn(x, (h0, c0))
        x, _ = self.rnn(x, h0)
        x = self.fc(x)
        x = self.out(x)
        x = x.squeeze(2)
        return x
"""


class TransformerClassification(nn.Module):
    def __init__(self, input_vector_size, output_vector_size):
        super(TransformerClassification, self).__init__()
        self.input_vector_size = input_vector_size
        self.output_vector_size = output_vector_size
        self.transformer_stage_1 = TransformerLayer(
            input_vector_size).to(DEVICE)
        self.transformer_stage_2 = TransformerLayer(
            input_vector_size).to(DEVICE)
        self.transformer_stage_3 = TransformerLayer(
            input_vector_size).to(DEVICE)

        input_length = time_to_index(STAGE_1_DURATION) + time_to_index(
            STAGE_2_DURATION) + time_to_index(STAGE_3_DURATION)
        self.fc1 = nn.Linear(input_length, output_vector_size*2)
        self.out = nn.Linear(output_vector_size*2, output_vector_size)

    def forward(self, stage_1, stage_2, stage_3):
        stage_1 = stage_1.to(DEVICE)
        stage_2 = stage_2.to(DEVICE)
        stage_3 = stage_3.to(DEVICE)

        stage_1_result = self.transformer_stage_1(stage_1)
        stage_2_result = self.transformer_stage_2(stage_2)
        stage_3_result = self.transformer_stage_3(stage_3)

        x = torch.cat((stage_1_result, stage_2_result, stage_3_result), dim=1)
        x = self.fc1(x)
        x = self.out(x)
        x = F.softmax(x, dim=1)
        return x


model = TransformerClassification(CHANNELS, N_CLASSES).to(DEVICE)
print(model)

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
    sample_array = sample_array.transpose()
    count = len([i for i in segmentations if i is not None])  # 计算划分点数
    if count == 2:
        stage_1 = sample_array[:seg_index[0], :]
        stage_2 = sample_array[seg_index[0]:seg_index[1], :]
        stage_3 = sample_array[seg_index[1]:, :]
    elif count == 1:
        assert seg_index[0] is not None
        stage_1 = sample_array[:seg_index[0], :]
        stage_2 = sample_array[seg_index[0]:, :]
        stage_3 = np.zeros((0, CHANNELS))
    elif count == 0:
        stage_1 = sample_array
        stage_2 = np.zeros((0, CHANNELS))
        stage_3 = np.zeros((0, CHANNELS))
    else:
        raise ValueError("Invalid segmentation points")

    def parse(t, sec=TIME_SERIES_DURATION):
        threshhold = time_to_index(sec)
        # 超长序列截断,超短序列补0
        if t.shape[0] > threshhold:
            t = t[: threshhold, :]
        else:
            t = np.pad(t, ((0,  threshhold - t.shape[0]), (0, 0)), 'constant')
        assert t.shape[0] == threshhold
        assert t.shape[1] == CHANNELS
        return t

    stage_1 = parse(stage_1, STAGE_1_DURATION)
    stage_2 = parse(stage_2, STAGE_2_DURATION)
    stage_3 = parse(stage_3, STAGE_3_DURATION)

    if batch_simulation:
        stage_1 = stage_1[np.newaxis, :, :]
        stage_2 = stage_2[np.newaxis, :, :]
        stage_3 = stage_3[np.newaxis, :, :]

    #print(stage_1.shape, stage_2.shape, stage_3.shape)
    return stage_1, stage_2, stage_3


class Dataset(Dataset):
    def __init__(self, dataset_length):
        self.y = []  # 目标值
        self.stages_1 = []
        self.stages_2 = []
        self.stages_3 = []
        self.generate_dataset(dataset_length)

    def generate_dataset(self, num):
        with alive_bar(num, title="数据集生成中") as bar:
            while len(self.stages_1) < num:  # 生成num个数据
                assert len(self.stages_1) == len(
                    self.stages_2) == len(self.stages_3) == len(self.y)
                sample_type = random.choice(SUPPORTED_SAMPLE_TYPES)
                sample, segmentations = get_sample(sample_type)  # 生成样本
                stage_1, stage_2, stage_3 = model_input_parse(
                    sample,
                    segmentations,
                    batch_simulation=False
                )  # 转换为模型输入格式
                self.stages_1.append(stage_1.astype(np.float32))
                self.stages_2.append(stage_2.astype(np.float32))
                self.stages_3.append(stage_3.astype(np.float32))
                # one-hot encoding
                index = SUPPORTED_SAMPLE_TYPES.index(sample_type)
                self.y.append([0] * index + [1] + [0] *
                              (len(SUPPORTED_SAMPLE_TYPES) - index - 1))
                bar()  # 进度条+1

    def __len__(self):
        assert len(self.stages_1) == len(
            self.stages_2) == len(self.stages_3) == len(self.y)
        return len(self.y)

    def __getitem__(self, idx):
        x = self.stages_1[idx], self.stages_2[idx], self.stages_3[idx]
        y = self.y[idx]
        return x, np.array(y)


def train():
    dataset = Dataset(DATASET_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True)
    for epoch in range(EPOCHS):  # 训练EPOCHS轮
        for i, (x, y) in enumerate(loader):
            stage_1, stage_2, stage_3 = x
            y = y.float().to(DEVICE)
            # print(stage_1)
            #print(stage_1.shape, stage_2.shape, stage_3.shape)
            optimizer.zero_grad()  # 梯度清零
            output = model(stage_1, stage_2, stage_3)  # 前向传播
            l = loss(output, y)  # 计算损失
            l.backward()  # 反向传播
            optimizer.step()  # 更新参数
            print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, i, l.item()))
    torch.save(model, FILE_PATH)  # 保存模型


def predict_raw_input(stage_1, stage_2, stage_3):
    assert os.path.exists(FILE_PATH), "model not found，please train first"
    model = torch.load(FILE_PATH, map_location=DEVICE).to(DEVICE)  # 加载模型
    # 转tensor
    stage_1 = torch.tensor(stage_1, dtype=torch.float32).to(DEVICE)
    stage_2 = torch.tensor(stage_2, dtype=torch.float32).to(DEVICE)
    stage_3 = torch.tensor(stage_3, dtype=torch.float32).to(DEVICE)
    model.eval()  # 验证模式
    with torch.no_grad():
        output = model(stage_1, stage_2, stage_3)
    return output


def predict(sample, segmentations=None):
    stage_1, stage_2, stage_3 = model_input_parse(
        sample,
        segmentations,
        batch_simulation=True
    )  # 转换为模型输入格式
    output = predict_raw_input(stage_1, stage_2, stage_3)
    return output.squeeze()


def test(type=None):
    if type is None:
        type = random.choice(SUPPORTED_SAMPLE_TYPES)
    sample, segmentations = get_sample(type)  # 生成样本
    output = predict(sample, segmentations)
    result_pretty = parse_predict_result(output)  # 解析结果
    print(result_pretty)
    label = get_label_from_result_pretty(result_pretty)  # 获取结果
    print(type, label)
    return label == type


if __name__ == "__main__":

    # train()
    test_cycle = 200
    accuracy = sum([test() for _ in range(test_cycle)])/test_cycle
    print("accuracy: {}".format(accuracy))
