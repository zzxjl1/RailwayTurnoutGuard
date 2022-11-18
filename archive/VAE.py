import random
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from sensor import SAMPLE_RATE, SUPPORTED_SAMPLE_TYPES, get_sample

FORCE_CPU = False  # 强制使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() and not FORCE_CPU
                      else 'cpu')
print('Using device:', DEVICE)
if DEVICE.type == 'cuda':
    torch.cuda.set_device(0)
    print('Using GPU:', torch.cuda.get_device_name(0))

EPOCHS = 100  # 训练数据集的轮次
LEARNING_RATE = 1e-4  # 学习率
BATCH_SIZE = 64  # 每批处理的数据

N_CLASSES = 12  # 分类数
DATASET_LENGTH = 50  # 数据集总长度
TIME_SERIES_DURATION = 20  # 20s

TIME_SERIES_LENGTH = SAMPLE_RATE * TIME_SERIES_DURATION  # 采样率*时间，总共的数据点数
SERIES_TO_ENCODE = ['A', 'B', 'C']  # 生成三相电流序列，不生成power曲线
POOLING_FACTOR_PER_TIME_SERIES = 10  # 每个时间序列的池化因子,用于降低工作量


def loss_fn(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


class VAEGT(nn.Module):
    def __init__(self, in_dims=784, hid1_dims=512, hid2_dims=256, num_classes=10, negative_slope=0.1):
        super(VAEGT, self).__init__()
        self.in_dims = in_dims
        self.hid1_dims = hid1_dims
        self.hid2_dims = hid2_dims
        self.num_classes = num_classes
        self.negative_slope = negative_slope

        # Encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('layer1', nn.Linear(in_dims, 2048)),
            ('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer2', nn.Linear(2048, 1024)),
            ('relu2', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer3', nn.Linear(1024, 768)),
            ('relu3', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
        ]))
        self.fc_mu = nn.Linear(768, hid1_dims)
        self.fc_var = nn.Linear(768, hid1_dims)

        # Conditioner
        self.conditioner = nn.Sequential(OrderedDict([
            ('layer1', nn.Linear(num_classes, 64)),
            ('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer2', nn.Linear(64, 128)),
            ('relu2', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer3', nn.Linear(128, hid2_dims)),
            ('relu3', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
        ]))

        # Decoder
        self.decoder = nn.Sequential(OrderedDict([
            ('layer1', nn.Linear(hid1_dims+hid2_dims, 512)),
            ('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer2', nn.Linear(512, 1024)),
            ('relu2', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer3', nn.Linear(1024, 2048)),
            ('relu3', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer4', nn.Linear(2048, in_dims)),
            ('sigmoid', nn.Sigmoid()),
        ]))

        self._init_weights()

    def forward(self, x, y):
        if self.training:
            # Encode input
            h = self.encoder(x)
            mu, logvar = self.fc_mu(h), self.fc_var(h)
            hx = self._reparameterize(mu, logvar)
            # Encode label
            y_onehot = y
            hy = self.conditioner(y_onehot)
            # Hidden representation
            h = torch.cat([hx, hy], dim=1)
            # Decode
            y = self.decoder(h)
            return y, mu, logvar
        else:
            hx = self._represent(x)
            hy = self.conditioner(self._onehot(y))
            h = torch.cat([hx, hy], dim=1)
            y = self.decoder(h)
            return y

    def generate(self, y):
        hy = self.conditioner(self._onehot(y))
        hx = self._sample(1).type_as(hy)
        h = torch.cat([hx, hy], dim=1)
        y = self.decoder(h)
        return y[0]

    def _represent(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        hx = self._reparameterize(mu, logvar)
        return hx

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).type_as(mu)
        z = mu + std * esp
        return z

    def _onehot(self, y):
        # 如果为int类型，转换为tensor
        if isinstance(y, int):
            y = torch.tensor([y], device=DEVICE)
            y = y.unsqueeze(0)
        y_onehot = torch.FloatTensor(y.shape[0], self.num_classes).to(DEVICE)
        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)
        return y_onehot

    def _sample(self, num_samples):
        return torch.FloatTensor(num_samples, self.hid1_dims).normal_()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


model = VAEGT(in_dims=TIME_SERIES_LENGTH*len(SERIES_TO_ENCODE) //
              POOLING_FACTOR_PER_TIME_SERIES, num_classes=N_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train(dataloader):
    model.train()
    for epoch in range(EPOCHS):
        for i, (timeseries, y_onehot) in enumerate(dataloader):
            batch_size = timeseries.shape[0]

            inputs = timeseries.view(batch_size, -1)
            #draw(inputs[0].detach().numpy(), "REAL")
            inputs = inputs.to(DEVICE)

            y_onehot = y_onehot.to(DEVICE)

            # Train
            optimizer.zero_grad()
            outputs, mu, logvar = model(inputs, y_onehot)
            loss = loss_fn(outputs, inputs, mu, logvar)
            loss.backward()
            optimizer.step()

        print("[EPOCH %.3d] Loss: %.6f" % (epoch, loss.item()))


def draw(y, title=""):
    plt.plot(y)
    TIME_SERIES_LENGTH = len(y)//3
    # x轴n等分，画竖线
    for i in range(len(SERIES_TO_ENCODE)+1):
        plt.axvline(x=TIME_SERIES_LENGTH*(i), color='r')
    plt.axvline(x=TIME_SERIES_LENGTH, color='r', linestyle='--')
    plt.title(title)
    plt.show()


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
        result = result[::POOLING_FACTOR_PER_TIME_SERIES]
        time_series.append(result)
    result = np.array(time_series)
    return result


def generate_dataset():
    """生成数据集"""
    x, y = [], []
    for _ in range(DATASET_LENGTH):
        type = random.choice(SUPPORTED_SAMPLE_TYPES)
        time_series = get_sample(type)
        # draw(time_series[::POOLING_FACTOR_PER_TIME_SERIES])
        x.append(time_series)
        index = SUPPORTED_SAMPLE_TYPES.index(type)
        # one-hot
        y.append([0]*index + [1] + [0]*(len(SUPPORTED_SAMPLE_TYPES)-index-1))
    return np.array(x), np.array(y)


def get_dataloader():
    DATASET = generate_dataset()
    x, y = map(lambda a: torch.tensor(np.array(a), dtype=torch.float,
               requires_grad=True), DATASET)  # 转换为tensor

    x, y = x.to(DEVICE), y.to(DEVICE)
    print("dataset shape", x.shape, y.shape)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    dataloader = get_dataloader()
    train(dataloader)

    for _ in range(2):
        model.eval()
        result = model.generate(0).cpu().detach().numpy()
        print(result.shape)
        draw(result, "Generated")
