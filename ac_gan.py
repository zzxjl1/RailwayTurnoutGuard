"""
一种修改过的ACGAN,将2d卷积换为1d，用于扩充数据集，改善数据集不平衡、数量不足的问题。
"""
import random
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import torch.nn as nn

from sensor import SAMPLE_RATE, SUPPORTED_SAMPLE_TYPES, generate_sample


BATCH_SIZE = 64  # 每批处理的数据
FORCE_CPU = True  # 强制使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() and not FORCE_CPU
                      else 'cpu')
print('Using device:', DEVICE)
EPOCHS = 100  # 训练数据集的轮次
LEARNING_RATE = 1e-3  # 学习率

N_CLASSES = 12
Z_DIM = 100


class Generator(nn.Module):
    '''
    pure Generator structure
    '''

    def __init__(self, n_classes, channels=1):

        super(Generator, self).__init__()
        self.channels = channels
        self.label_embedding_dim = Z_DIM
        self.n_classes = n_classes

        self.label_embedding = nn.Embedding(
            self.n_classes, self.label_embedding_dim)
        self.linear = nn.Linear(self.label_embedding_dim, 768)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=768, out_channels=384, kernel_size=4, stride=1,
                               padding=0, bias=False),
            nn.BatchNorm1d(384),
            nn.ReLU(True)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=384, out_channels=256, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=256, out_channels=192, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(True),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=192, out_channels=64, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )

        self.last = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=self.channels, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_emb = self.label_embedding(labels)
        gen_input = torch.mul(label_emb, z)

        out = self.linear(gen_input)
        out = out.view(-1, 768, 1, 1)

        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)

        out = self.last(out)  # (*, c, 64, 64)

        return out


class Discriminator(nn.Module):
    '''
    pure discriminator structure
    '''

    def __init__(self, n_classes, channels=1):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.n_classes = n_classes

        # (*, c, 64, 64)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.channels, out_channels=16,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )

        # (*, 64, 32, 32)
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )

        # (*, 128, 16, 16)
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )

        # (*, 256, 8, 8)
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )

        # output layers
        # (*, 512, 8, 8)
        # dis fc
        self.last_adv = nn.Sequential(
            nn.Linear(450*512, 1),
            nn.Sigmoid()
        )
        # aux classifier fc
        self.last_aux = nn.Sequential(
            nn.Linear(450*512, self.n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)

        flat = out.view(input.size(0), -1)

        fc_dis = self.last_adv(flat)
        fc_aux = self.last_aux(flat)

        return fc_dis.squeeze(), fc_aux


def weights_init(m):
    '''
    custom weights initializaiont called on G and D, from the paper DCGAN
    Args:
        m (tensor): the network parameters
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


generator = Generator(n_classes=N_CLASSES).to(DEVICE)
discriminator = Discriminator(n_classes=N_CLASSES).to(DEVICE)

generator.apply(weights_init)
discriminator.apply(weights_init)

# optimizer
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

# for orignal gan loss function
adversarial_loss_sigmoid = nn.BCEWithLogitsLoss()
aux_loss = nn.CrossEntropyLoss()


def tensor2var(x):
    x = x.to(DEVICE)
    return x


def compute_acc(real_aux, fake_aux, labels, gen_labels):
    # Calculate discriminator accuracy
    pred = np.concatenate([real_aux.data.cpu().numpy(),
                          fake_aux.data.cpu().numpy()], axis=0)
    gt = np.concatenate(
        [labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
    d_acc = np.mean(np.argmax(pred, axis=1) == gt)

    return d_acc


def train(data_loader):

    for epoch in range(EPOCHS):

        for i, (real_timeseries, labels) in enumerate(data_loader):
            # configure input
            real_timeseries = tensor2var(real_timeseries)
            print(real_timeseries.shape)
            labels = tensor2var(labels)

            # adversarial ground truths
            valid = tensor2var(torch.full(
                (real_timeseries.size(0),), 0.9))  # (*, )
            fake = tensor2var(torch.full(
                (real_timeseries.size(0),), 0.0))  # (*, )

            # ==================== Train D ==================
            discriminator.train()
            generator.train()

            discriminator.zero_grad()

            # compute loss with real images
            dis_out_real, aux_out_real = discriminator(real_timeseries)

            d_loss_real = adversarial_loss_sigmoid(
                dis_out_real, valid) + aux_loss(aux_out_real, labels)

            # noise z for generator
            z = tensor2var(torch.randn(
                real_timeseries.size(0), Z_DIM))  # *, 100
            gen_labels = tensor2var(torch.randint(
                0, N_CLASSES, (real_timeseries.size(0),), dtype=torch.long))

            fake_timeseries = generator(z, gen_labels)  # (*, c, 64, 64)
            dis_out_fake, aux_out_fake = discriminator(fake_timeseries)  # (*,)

            d_loss_fake = adversarial_loss_sigmoid(
                dis_out_fake, fake) + aux_loss(aux_out_fake, gen_labels)

            # total d loss
            d_loss = d_loss_real + d_loss_fake

            d_loss.backward()
            # update D
            discriminator_optimizer.step()

            # calculate dis accuracy
            d_acc = compute_acc(aux_out_real, aux_out_fake, labels, gen_labels)

            # train the generator every 5 steps
            if i % 5 == 0:

                # =================== Train G and gumbel =====================
                generator.zero_grad()
                # create random noise
                fake_timeseries = generator(z, gen_labels)

                # compute loss with fake images
                dis_out_fake, aux_out_fake = discriminator(
                    fake_timeseries)  # batch x n

                g_loss_fake = adversarial_loss_sigmoid(
                    dis_out_fake, valid) + aux_loss(aux_out_fake, gen_labels)

                g_loss_fake.backward()
                # update G
                generator_optimizer.step()

        # end one epoch

        print("epoch:{}, d_loss: {:.4f}, g_loss: {:.4f}, d_acc: {:.4f}"
              .format(epoch, d_loss.item(), g_loss_fake.item(), d_acc))


TIME_SERIES_DURATION = 20
TIME_SERIES_LENGTH = SAMPLE_RATE * TIME_SERIES_DURATION
SERIES_TO_ENCODE = ['A', 'B', 'C']


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
    temp, _ = generate_sample(type=type)
    time_series = []
    for type in SERIES_TO_ENCODE:
        result = parse(temp[type][1])
        time_series += list(result)  # concat操作
    return time_series


def generate_dataset():
    """生成数据集"""
    DATASET_LENGTH = 100  # 数据集总长度
    x, y = [], []
    for _ in range(DATASET_LENGTH):
        type = random.choice(SUPPORTED_SAMPLE_TYPES)
        time_series = get_sample(type)
        x.append([time_series])
        index = SUPPORTED_SAMPLE_TYPES.index(type)
        y.append([0] * index + [1] + [0] *
                 (len(SUPPORTED_SAMPLE_TYPES) - index - 1))
    return x, y


def get_dataloader():
    DATASET = generate_dataset()

    x, y = map(lambda a: torch.tensor(np.array(a), dtype=torch.float,
               requires_grad=True), DATASET)  # 转换为tensor

    x, y = x.to(DEVICE), y.to(DEVICE)
    print(x.shape, y.shape)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)


def draw(y, title=""):
    plt.plot(y)
    # x轴n等分，画竖线
    for i in range(len(SERIES_TO_ENCODE)+1):
        plt.axvline(x=TIME_SERIES_LENGTH*(i), color='r')
    plt.axvline(x=TIME_SERIES_LENGTH, color='r', linestyle='--')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    dataloader = get_dataloader()
    train(dataloader)
