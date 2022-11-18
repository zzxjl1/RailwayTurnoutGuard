from time import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sensor import SAMPLE_RATE, SUPPORTED_SAMPLE_TYPES, get_sample

FORCE_CPU = False  # 强制使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() and not FORCE_CPU
                      else 'cpu')
print('Using device:', DEVICE)
if DEVICE.type == 'cuda':
    torch.cuda.set_device(1)
    print('Using GPU:', torch.cuda.get_device_name(1))

EPOCHS = 10000  # 训练数据集的轮次
LEARNING_RATE = 1e-4  # 学习率
BATCH_SIZE = 64  # 每批处理的数据

N_CLASSES = 12  # 分类数
NOISE_DIM = 20  # 噪声维度
DATASET_LENGTH = 500  # 数据集总长度
TIME_SERIES_DURATION = 20  # 20s

TIME_SERIES_LENGTH = SAMPLE_RATE * TIME_SERIES_DURATION  # 采样率*时间，总共的数据点数
SERIES_TO_ENCODE = ['A', 'B', 'C']  # 生成三相电流序列，不生成power曲线
POOLING_FACTOR_PER_TIME_SERIES = 10  # 每个时间序列的池化因子,用于降低工作量


class _Block(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.5
    ):
        super(_Block, self).__init__()

        self.padding = padding
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.resample = (
            nn.Conv1d(n_inputs, n_outputs,
                      1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.fill_(0.05)
        self.conv2.weight.data.fill_(0.05)
        if self.resample != None:
            self.resample.weight.data.fill_(0.05)

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = x0[:, :, : -self.padding].contiguous()
        x0 = self.relu1(x0)
        x0 = self.dropout1(x0)

        x0 = self.conv2(x0)
        x0 = x0[:, :, : -self.padding].contiguous()
        x0 = self.relu2(x0)
        out = self.dropout2(x0)

        res = x if self.resample is None else self.resample(x)
        return self.relu(out + res)


class CausalDialationBlock(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.5):
        super(CausalDialationBlock, self).__init__()
        layers = []
        num_levels = len(num_channels)

        layers = [
            _Block(
                num_inputs,
                num_channels[0],
                kernel_size,
                stride=1,
                dilation=1,
                padding=kernel_size - 1,
                dropout=dropout,
            )
        ]
        layers += [
            _Block(
                num_channels[i - 1],
                num_channels[i],
                kernel_size,
                stride=1,
                dilation=((2 ** i) % 512),
                padding=(kernel_size - 1) * ((2 ** i) % 512),
                dropout=dropout,
            )
            for i in range(1, num_levels)
        ]

        self.network = nn.Sequential(*layers)
        self.lstm = nn.LSTM(num_inputs, 256, 1, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(256, 1), nn.Tanh())

    def forward(self, x):
        return self.network(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        in_dim,
        kernel_size=8,
        n_layers=1,
        hidden_dim=256,
        n_channel=3,
        cnn_layers=4,
        dropout=0.5,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear_1 = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

        num_channels = [n_channel] * cnn_layers

        self.causalBlock = CausalDialationBlock(
            in_dim, num_channels, kernel_size=kernel_size, dropout=dropout
        )
        self.rectify = nn.ReLU()
        self.linear_2 = nn.Linear(num_channels[-1], 1)
        self.linear_2.weight.data.fill_(0.01)

    def forward(self, input, channel_last=True):

        # print(input.shape)
        # 降采样
        input = input[:, ::POOLING_FACTOR_PER_TIME_SERIES, :]

        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(self.n_layers, batch_size,
                          self.hidden_dim).to(DEVICE)
        c_0 = torch.zeros(self.n_layers, batch_size,
                          self.hidden_dim).to(DEVICE)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear_1(
            recurrent_features.contiguous().view(batch_size * seq_len, self.hidden_dim)
        )
        outputs = outputs.view(batch_size, seq_len, 1)

        y1 = self.causalBlock(outputs.transpose(
            1, 2) if channel_last else outputs)

        j = self.linear_2(y1.transpose(1, 2))
        j0 = self.rectify(j)
        r = torch.tanh(j0)
        return r


class Generator(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        n_channel,
        kernel_size,
        dropout=0.5,
        n_layers=1,
        hidden_dim=256,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear_1 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim), nn.Tanh())
        num_channels = [n_channel] * n_layers

        self.causalBlock = CausalDialationBlock(
            in_dim, num_channels, kernel_size=kernel_size, dropout=dropout
        )
        self.linear_2 = nn.Linear(num_channels[-1], 1)
        self.linear_2.weight.data.fill_(0.5)

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)

        h_0 = torch.zeros(self.n_layers, batch_size,
                          self.hidden_dim).to(DEVICE)
        c_0 = torch.zeros(self.n_layers, batch_size,
                          self.hidden_dim).to(DEVICE)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))

        y1 = self.causalBlock(
            recurrent_features.transpose(1, 2)
        )

        j = self.linear_2(y1.transpose(1, 2))

        r = torch.relu(j)
        r = r.reshape(batch_size, seq_len, 1)

        return r


netD = Discriminator(
    in_dim=1,
    cnn_layers=3,
    n_layers=1,
    kernel_size=8,
    n_channel=2,
    hidden_dim=NOISE_DIM,
).to(DEVICE)

netG = Generator(
    in_dim=NOISE_DIM,
    n_channel=2,
    kernel_size=8,
    out_dim=1,
    hidden_dim=NOISE_DIM
).to(DEVICE)

criterion = nn.BCELoss().to(DEVICE)


fixed_noise = torch.randn(
    BATCH_SIZE, TIME_SERIES_LENGTH, NOISE_DIM, device=DEVICE)

fake_label = 0.0
real_label = 1.0

optimizerD = torch.optim.Adam(netD.parameters(), lr=LEARNING_RATE)
optimizerG = torch.optim.Adam(netG.parameters(), lr=LEARNING_RATE)
torch.manual_seed(0)


def train(dataloader):
    for epoch in range(EPOCHS):
        for i, data in enumerate(dataloader):
            data = data[0].to(DEVICE)
            batch_size, seq_len = data.size(0), data.size(1)

            netD.zero_grad()
            real = data
            # 加一个维度
            real = real.unsqueeze(2)
            output = netD(real)
            label = torch.full((batch_size, output.size(1), 1),
                               real_label, device=DEVICE)

            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(batch_size, seq_len, NOISE_DIM, device=DEVICE)
            fake = netG(noise)

            label.fill_(fake_label)
            output = netD(fake.detach())

            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake)

            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()

            print(
                f"[{epoch}/{EPOCHS}][{i}/{len(dataloader)}] Loss_D:{errD.item():.4f} Loss_D_REAL:{errD_real.item():.4f} Loss_D_FAKE:{errD_fake.item():.4f} Loss_G:{errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
            )
            # if epoch%5==0 and i==0:
            #    test_generate()


def test_generate():
    t = generate()
    draw(t)


def parse(time_series):
    # 超长的截断，短的补0
    if len(time_series) > TIME_SERIES_LENGTH:
        return np.array(
            time_series[:TIME_SERIES_LENGTH])
    else:
        return np.pad(
            time_series, (0, TIME_SERIES_LENGTH - len(time_series)), 'constant')


def get_sample(type):
    """获取拼接后的时间序列，比如Phase A, B, C连在一起，这样做是为了输入模型中"""
    temp, _ = get_sample(type=type)
    time_series = []
    for type in SERIES_TO_ENCODE:
        result = parse(temp[type][1])
        time_series += list(result)
    result = np.array(time_series)
    return result


def generate_dataset(type):
    """生成数据集"""
    x = []
    for _ in range(DATASET_LENGTH):
        time_series = get_sample(type)
        # draw(time_series[::POOLING_FACTOR_PER_TIME_SERIES])
        x.append(time_series)
    return np.array(x)


def get_dataloader(type="normal"):
    DATASET = generate_dataset(type)
    t = torch.tensor(DATASET, dtype=torch.float,
                     requires_grad=True)  # 转换为tensor

    t = t.to(DEVICE)
    print("dataset shape", t.shape)
    ds = TensorDataset(t)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)


def generate():
    target_length = TIME_SERIES_LENGTH*3//POOLING_FACTOR_PER_TIME_SERIES
    noise = torch.randn(1, target_length, NOISE_DIM, device=DEVICE)
    fake = netG(noise)
    result = fake.reshape(target_length)
    return result.cpu().detach().numpy()


def draw(y, title=""):
    plt.plot(y)
    TIME_SERIES_LENGTH = len(y)//3
    # x轴n等分，画竖线
    for i in range(len(SERIES_TO_ENCODE)+1):
        plt.axvline(x=TIME_SERIES_LENGTH*(i), color='r')
    plt.axvline(x=TIME_SERIES_LENGTH, color='r', linestyle='--')
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    dataloader = get_dataloader()
    train(dataloader)

    t = generate()
    draw(t)
