"""
训练用GRU对时间序列上的每一点作为分割点的可能性进行预测
这个模型的输出会用于和原文中的分割算法结合，提升分割的准确性
RNN对序列敏感，因此能够捕捉到stage切换间的变化
"""
import os
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sensor.config import SAMPLE_RATE, SUPPORTED_SAMPLE_TYPES
from sensor.dataset import generate_dataset


class GRUScore(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, num_layers=1, dropout=0):
        super(GRUScore, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

        self.init_weights()

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(
            self.num_layers, x.shape[0], self.hidden_size).to(self.device)
        # Forward propagate RNN
        out, _ = self.gru(x, h0)
        # Decode the hidden state of the last time step
        out = self.fc(out)
        out = self.activation(out)
        return out

    # 权重初始化
    def init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                param.data[self.hidden_size:2 * self.hidden_size] = 1
        for name, param in self.fc.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        for name, param in self.activation.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


FILE_PATH = "./models/gru_score.pth"
DATASET_LENGTH = 100  # 数据集总长度
TIME_SERIES_DURATION = 20  # 20s
TIME_SERIES_LENGTH = SAMPLE_RATE * TIME_SERIES_DURATION  # 采样率*时间，总共的数据点数
SERIES_TO_ENCODE = ['A', 'B', 'C']  # 生成三相电流序列，不生成power曲线
POOLING_FACTOR_PER_TIME_SERIES = 5  # 每个时间序列的池化因子,用于降低工作量

EPOCHS = 200  # 训练数据集的轮次
LEARNING_RATE = 1e-3  # 学习率
BATCH_SIZE = 64  # 每批处理的数据
FORCE_CPU = True  # 强制使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available()
                      and not FORCE_CPU else 'cpu')
CHANNELS = len(SERIES_TO_ENCODE)  # 通道数
TRAIN_ONLY_WITH_NORMAL = True  # 只用正常数据训练（！经测试，使用故障样本训练会无法收敛！）


def get_dataloader():
    x, seg_indexs = generate_dataset(DATASET_LENGTH, TIME_SERIES_LENGTH, type="normal" if TRAIN_ONLY_WITH_NORMAL else None,
                                     pooling_factor_per_time_series=POOLING_FACTOR_PER_TIME_SERIES, series_to_encode=SERIES_TO_ENCODE)
    dataset_length, channels, seq_len = x.shape
    x = x.transpose(0, 2, 1)
    x = torch.from_numpy(x).float()
    # y的shape是(数据集长度,时间序列长度,1)
    y = torch.zeros((DATASET_LENGTH, seq_len)).float()
    for i in range(DATASET_LENGTH):
        seg_index = [x for x in seg_indexs[i] if x is not None]
        y[i, seg_index] = 1
        # print(seg_index[i])
    y = y.unsqueeze(2)
    print(x.shape, y.shape)

    # print(list(y[0].detach().cpu().numpy()))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


model = GRUScore(input_size=CHANNELS, hidden_size=TIME_SERIES_LENGTH //
                 POOLING_FACTOR_PER_TIME_SERIES, output_size=1, device="cpu")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train():
    dataloader = get_dataloader()
    for epoch in range(EPOCHS):
        for i, (x, y) in enumerate(dataloader):
            # print(x.shape, y.shape)
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, EPOCHS, i + 1, len(dataloader), loss.item()))

    torch.save(model, FILE_PATH)


def predict(t) -> np.ndarray:
    batch_size, channels, seq_len = t.shape
    assert channels == CHANNELS
    assert os.path.exists(FILE_PATH), "please train() first"
    model = torch.load(FILE_PATH)
    input = t.transpose(0, 2, 1)
    input = torch.from_numpy(input).float()
    out = model(input)
    out = out.detach().cpu().numpy()
    return out


def test(type="normal"):
    t, seg_indexs = generate_dataset(dataset_length=1,
                                     time_series_length=TIME_SERIES_LENGTH,
                                     type=type,
                                     pooling_factor_per_time_series=POOLING_FACTOR_PER_TIME_SERIES,
                                     series_to_encode=SERIES_TO_ENCODE)
    out = predict(t)
    fig = plt.figure(dpi=150, figsize=(9, 2))
    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    #ax2.plot(out.squeeze(), label="score")
    # 热度图
    ax1.pcolormesh(out.reshape(1, -1), cmap="Reds", alpha=0.8)
    sample = t.squeeze()
    for i in range(len(SERIES_TO_ENCODE)):
        ax2.plot(sample[i], label=SERIES_TO_ENCODE[i])
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='best')  # 显示图例
    plt.title("Segmentation GRU Score Heatmap - type: {}".format(type))
    ax1.set_yticks([])  # 不显示y轴
    ax2.yaxis.tick_left()  # ax2在左边
    plt.show()


if __name__ == "__main__":
    train()
    for type in SUPPORTED_SAMPLE_TYPES:
        test(type)
