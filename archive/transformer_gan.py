from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from sensor import SAMPLE_RATE, SUPPORTED_SAMPLE_TYPES, get_sample

FORCE_CPU = False  # 强制使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() and not FORCE_CPU
                      else 'cpu')
print('Using device:', DEVICE)
if DEVICE.type == 'cuda':
    torch.cuda.set_device(0)
    print('Using GPU:', torch.cuda.get_device_name(0))

EPOCHS = 1000  # 训练数据集的轮次
LEARNING_RATE = 1e-4  # 学习率
BATCH_SIZE = 64  # 每批处理的数据

N_CLASSES = 1  # 分类数
NOISE_DIM = 10  # 噪声维度
DATASET_LENGTH = 500  # 数据集总长度
TIME_SERIES_DURATION = 20  # 20s

TIME_SERIES_LENGTH = SAMPLE_RATE * TIME_SERIES_DURATION  # 采样率*时间，总共的数据点数
SERIES_TO_ENCODE = ['A', 'B', 'C']  # 生成三相电流序列，不生成power曲线
POOLING_FACTOR_PER_TIME_SERIES = 10  # 每个时间序列的池化因子,用于降低工作量


class Generator(nn.Module):
    def __init__(self, seq_len, channels, patch_size=40, embed_dim=10, depth=5,
                 forward_drop_rate=0.5, attn_drop_rate=0.5):
        super(Generator, self).__init__()
        self.channels = channels
        self.NOISE_DIM = NOISE_DIM
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate

        self.l1 = nn.Linear(self.NOISE_DIM, self.seq_len * self.embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, self.embed_dim))
        self.blocks = Gen_TransformerEncoder(
            depth=self.depth,
            emb_size=self.embed_dim,
            drop_p=self.attn_drop_rate,
            forward_drop_p=self.forward_drop_rate
        )

        self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.channels, 1, 1, 0)
        )

    def forward(self, z):
        x = self.l1(z)
        x = x.view(-1, self.seq_len, self.embed_dim)
        x = x + self.pos_embed
        H, W = 1, self.seq_len
        x = self.blocks(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        output = self.deconv(x.permute(0, 3, 1, 2))
        output = output.view(-1, self.channels, H, W)
        return output

    def generate(self):
        z = torch.Tensor(np.random.normal(0, 1, (1, NOISE_DIM))).to(DEVICE)
        fake_timeseries = gen_net(z)
        result = fake_timeseries.squeeze().cpu().detach().numpy()
        return result


class Gen_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class Gen_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Gen_TransformerEncoderBlock(**kwargs)
                           for _ in range(depth)])


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(
            x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d",
                         h=self.num_heads)
        values = rearrange(self.values(
            x), "b n (h d) -> b h n d", h=self.num_heads)
        # batch, num_heads, query_len, key_len
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class Dis_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size=100,
                 num_heads=5,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class Dis_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Dis_TransformerEncoderBlock(**kwargs)
                           for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=100, n_classes=2):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return out


class PatchEmbedding_Linear(nn.Module):
    # what are the proper parameters set here?
    def __init__(self, in_channels=21, patch_size=16, emb_size=100, seq_length=1024):
        # self.patch_size = patch_size
        super().__init__()
        # change the conv2d parameters here
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)',
                      s1=1, s2=patch_size),
            nn.Linear(patch_size*in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn(
            (seq_length // patch_size) + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # position
        x += self.positions
        return x


class Discriminator(nn.Sequential):
    def __init__(self,
                 seq_length,
                 in_channels,
                 n_classes,
                 patch_size=15,
                 emb_size=50,
                 depth=3,
                 **kwargs):
        super().__init__(
            PatchEmbedding_Linear(in_channels, patch_size,
                                  emb_size, seq_length),
            Dis_TransformerEncoder(
                depth, emb_size=emb_size, drop_p=0.5, forward_drop_p=0.5, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )


gen_net = Generator(seq_len=TIME_SERIES_LENGTH//POOLING_FACTOR_PER_TIME_SERIES,
                    channels=len(SERIES_TO_ENCODE)).to(DEVICE)
dis_net = Discriminator(seq_length=TIME_SERIES_LENGTH//POOLING_FACTOR_PER_TIME_SERIES,
                        in_channels=len(SERIES_TO_ENCODE),
                        n_classes=N_CLASSES).to(DEVICE)

gen_optimizer = torch.optim.Adam(gen_net.parameters(), LEARNING_RATE)
dis_optimizer = torch.optim.Adam(dis_net.parameters(), LEARNING_RATE)


def train(train_loader):

    global_steps = 0

    gen_net.train()
    dis_net.train()

    dis_optimizer.zero_grad()
    gen_optimizer.zero_grad()

    for epoch in range(EPOCHS):

        for index, (timeseries, label) in enumerate(train_loader):
            batch_size = timeseries.shape[0]
            # 插入一个维度
            timeseries = timeseries.unsqueeze(2)

            #print(timeseries.shape, label.shape)

            # Adversarial ground truths
            real_timeseries = timeseries.to(DEVICE)

            # Sample noise as generator input
            z = torch.Tensor(np.random.normal(
                0, 1, (batch_size, NOISE_DIM))).to(DEVICE)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            real_validity = dis_net(real_timeseries)
            fake_timeseries = gen_net(z).detach()

            assert fake_timeseries.size() == real_timeseries.size(
            ), f"fake_timeseries.size(): {fake_timeseries.size()} real_timeseries.size(): {real_timeseries.size()}"

            fake_validity = dis_net(fake_timeseries)

            real_label = torch.full(
                (batch_size*N_CLASSES,), 0.9, dtype=torch.float, device=DEVICE)
            fake_label = torch.full(
                (batch_size*N_CLASSES,), 0.1, dtype=torch.float, device=DEVICE)
            real_validity = nn.Sigmoid()(real_validity.view(-1))
            fake_validity = nn.Sigmoid()(fake_validity.view(-1))

            d_real_loss = nn.BCELoss()(real_validity, real_label)
            d_fake_loss = nn.BCELoss()(fake_validity, fake_label)
            d_loss = d_real_loss + d_fake_loss

            d_loss.backward()

            torch.nn.utils.clip_grad_norm_(dis_net.parameters(), 5.)
            dis_optimizer.step()
            dis_optimizer.zero_grad()

            # -----------------
            #  Train Generator
            # -----------------
            if global_steps % 1 == 0:

                gen_z = torch.cuda.FloatTensor(np.random.normal(
                    0, 1, (batch_size, NOISE_DIM)))
                gen_timeseries = gen_net(gen_z)
                fake_validity = dis_net(gen_timeseries)

                real_label = torch.full(
                    (batch_size*N_CLASSES,), 1., dtype=torch.float, device=real_timeseries.get_device())
                fake_validity = nn.Sigmoid()(fake_validity.view(-1))
                g_loss = nn.BCELoss()(fake_validity, real_label)
                g_loss.backward()

                torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.)
                gen_optimizer.step()
                gen_optimizer.zero_grad()

            global_steps += 1
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] " %
                  (epoch, EPOCHS, index % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

            """if epoch%100==0 and index==0:
                draw(gen_net.generate().ravel())"""

    draw(gen_net.generate().ravel())


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


def generate_dataset(type):
    """生成数据集"""
    x, y = [], []
    for _ in range(DATASET_LENGTH):
        time_series = get_sample(type)
        # draw(time_series[::POOLING_FACTOR_PER_TIME_SERIES])
        x.append(time_series)
        index = SUPPORTED_SAMPLE_TYPES.index(type)
        y.append([index])
    return np.array(x), np.array(y)


def get_dataloader(type="normal"):
    DATASET = generate_dataset(type)
    x, y = map(lambda a: torch.tensor(np.array(a), dtype=torch.float,
               requires_grad=True), DATASET)  # 转换为tensor

    x, y = x.to(DEVICE), y.to(DEVICE)
    print("dataset shape", x.shape, y.shape)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)


if __name__ == "__main__":
    train_loader = get_dataloader()
    train(train_loader)
