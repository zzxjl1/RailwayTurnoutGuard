"""
一种修改过的ACGAN,将2d卷积换为1d，用于扩充数据集，改善数据集不平衡、数量不足的问题。
注意：这里的ACGAN是针对时间序列的，所以输入的数据是时间序列，而不是图片。
如果lstm-gan效果更好，则计划只用于baseline
"""
import random
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import torch.nn as nn
from sensor import SAMPLE_RATE, SUPPORTED_SAMPLE_TYPES, generate_sample


BATCH_SIZE = 32  # 每批处理的数据
FORCE_CPU = True  # 强制使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() and not FORCE_CPU
                      else 'cpu')
print('Using device:', DEVICE)
EPOCHS = 100  # 训练数据集的轮次
LEARNING_RATE = 1e-4  # 学习率

N_CLASSES = 12  # 分类数
NOISE_DIM = 100  # 噪声维度
DATASET_LENGTH = 500  # 数据集总长度
TIME_SERIES_DURATION = 20  # 20s
TIME_SERIES_LENGTH = SAMPLE_RATE * TIME_SERIES_DURATION  # 采样率*时间，总共的数据点数
SERIES_TO_ENCODE = ['A', 'B', 'C']  # 生成三相电流序列，不生成power曲线


class Generator(nn.Module):
    '''
    pure Generator structure
    '''

    def __init__(self, n_classes, channels=len(SERIES_TO_ENCODE)):

        super(Generator, self).__init__()
        self.channels = channels
        self.label_embedding_dim = NOISE_DIM
        self.n_classes = n_classes

        self.label_embedding = nn.Embedding(
            self.n_classes, self.label_embedding_dim)
        self.linear = nn.Linear(self.label_embedding_dim, 768)

        self.deconv1 = nn.Sequential(
            # (Lin​−1)×stride+kernel_size
            nn.ConvTranspose1d(in_channels=768, out_channels=384, kernel_size=32, stride=1,
                               padding=0, bias=False),
            nn.BatchNorm1d(384),
            nn.ReLU(True)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=384, out_channels=256, kernel_size=32, stride=1,
                               padding=0, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=256, out_channels=192, kernel_size=32, stride=2,
                               padding=0, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(True),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=192, out_channels=64, kernel_size=32, stride=2,
                               padding=0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )

        self.last = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=self.channels, kernel_size=32, stride=4,
                               padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_emb = self.label_embedding(labels)
        gen_input = torch.mul(label_emb, z)
        out = self.linear(gen_input)
        out = out.view(-1, 768, 1)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.last(out)

        # print("generater output shape before interpolte:", out.shape)  # debug only
        out = torch.nn.functional.interpolate(
            out, size=TIME_SERIES_LENGTH, mode='linear')
        return out

    def generate(self, sample_type):
        z = torch.randn(1, NOISE_DIM)
        label = torch.tensor(SUPPORTED_SAMPLE_TYPES.index(
            sample_type), dtype=torch.long)
        result = self.forward(z, label)
        result = result.detach().numpy()
        return result[0]


class Discriminator(nn.Module):
    '''
    pure discriminator structure
    '''

    def __init__(self, n_classes, channels=len(SERIES_TO_ENCODE)):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.n_classes = n_classes

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.channels, out_channels=16,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )

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

        # 置信度
        self.last_adv = nn.Sequential(
            nn.Linear(150*512, 1),
            nn.Sigmoid()
        )
        # 分类情况
        self.last_aux = nn.Sequential(
            nn.Linear(150*512, self.n_classes),
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
# print(generator)
# print(discriminator)

generator.apply(weights_init)  # 初始化权重
discriminator.apply(weights_init)  # 初始化权重

# optimizer
generator_optimizer = torch.optim.Adam(
    generator.parameters(), lr=LEARNING_RATE)
discriminator_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=LEARNING_RATE)

# 损失函数
adversarial_loss_sigmoid = nn.BCEWithLogitsLoss()
aux_loss = nn.CrossEntropyLoss()


def tensor2var(x):
    x = x.to(DEVICE)
    return x


def compute_acc(real_aux, fake_aux, labels, gen_labels):
    # 计算鉴别器的准确率
    pred = np.concatenate([real_aux.data.cpu().numpy(),
                          fake_aux.data.cpu().numpy()], axis=0)
    #print(labels, gen_labels)
    gt = np.concatenate(
        [labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
    d_acc = np.mean(np.argmax(pred, axis=1) == gt)

    return d_acc


def train(data_loader):

    for epoch in range(EPOCHS):

        for i, (real_timeseries, labels) in enumerate(data_loader):
            # configure input
            real_timeseries = tensor2var(real_timeseries)
            #print("real shape", real_timeseries.shape)
            labels = tensor2var(labels).long()

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
            noise = tensor2var(torch.randn(
                real_timeseries.size(0), NOISE_DIM))  # *, 100
            gen_labels = tensor2var(torch.randint(
                0, N_CLASSES, (real_timeseries.size(0),), dtype=torch.long))

            fake_timeseries = generator(noise, gen_labels)  # (*, c, 64, 64)
            #print("fake shape", fake_timeseries.shape)
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
                fake_timeseries = generator(noise, gen_labels)

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
        time_series.append(list(result))  # concat操作
    return time_series


def generate_dataset(use_onehot):
    """生成数据集"""
    x, y = [], []
    for _ in range(DATASET_LENGTH):
        type = random.choice(SUPPORTED_SAMPLE_TYPES)
        time_series = get_sample(type)
        x.append(time_series)
        index = SUPPORTED_SAMPLE_TYPES.index(type)
        if use_onehot:
            y.append(np.eye(N_CLASSES)[index])
        else:
            y.append(index)
    return x, y


def get_dataloader(use_onehot=False):
    DATASET = generate_dataset(use_onehot)

    x, y = map(lambda a: torch.tensor(np.array(a), dtype=torch.float,
               requires_grad=True), DATASET)  # 转换为tensor

    x, y = x.to(DEVICE), y.to(DEVICE)
    print("dataset shape", x.shape, y.shape)
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
    type = "normal"
    t = generator.generate(type)
    print(t, t.shape)
    draw(t.flatten(), title=type)
