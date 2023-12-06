"""
降噪自编码器 Denoising Auto-Encoder
采用正常时间序列无监督训练，用于产生是否异常的置信度
该置信度会用于之后的分类，以降低假阳率
"""
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sensor import (
    SAMPLE_RATE,
    SUPPORTED_SAMPLE_TYPES,
    generate_dataset,
    get_sample,
    parse_sample,
)
from gru_score import GRUScore
from torch.utils.tensorboard import SummaryWriter
from pytorchtools import EarlyStopping  # Add an EarlyStopping utility

writer = SummaryWriter("./paper/ae")  # TensorBoard writer
early_stopping = EarlyStopping(patience=20, verbose=True)


POOLING_FACTOR_PER_TIME_SERIES = 5  # 每条时间序列的降采样因子
TIME_SERIES_DURATION = 10  # 输入模型的时间序列时长为10s
TIME_SERIES_LENGTH = SAMPLE_RATE * TIME_SERIES_DURATION  # 时间序列长度
TRAINING_SET_LENGTH = 30  # 训练集长度
TESTING_SET_LENGTH = 10  # 测试集长度
SERIES_TO_ENCODE = ["A", "B", "C"]  # 参与训练和预测的序列，power暂时不用
CHANNELS = len(SERIES_TO_ENCODE)
TOTAL_LENGTH = TIME_SERIES_LENGTH // POOLING_FACTOR_PER_TIME_SERIES

MODEL_TO_USE = "BP"
LEARNING_RATE = 1e-4  # 学习率
BATCH_SIZE = 128  # 批大小
FILE_PATH = "./models/auto_encoder/"  # 模型保存路径
FORCE_CPU = False  # 强制使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
print("Using device:", DEVICE)


if MODEL_TO_USE == "BP":
    TOTAL_LENGTH *= CHANNELS  # 输入总长度
print("total input length:", TOTAL_LENGTH)


class BP_AE(nn.Module):
    def __init__(self, seq_len, latent_dim):
        super(BP_AE, self).__init__()
        self.bottle_neck_output = None

        self.encoder = nn.Sequential(
            nn.Linear(seq_len, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, seq_len),
        )

    def forward(self, x):
        x = self.encoder(x)
        self.bottle_neck_output = x
        x = self.decoder(x)
        return x


class EncoderRNN(nn.Module):
    def __init__(self, n_features, latent_dim, hidden_size):
        super(EncoderRNN, self).__init__()

        self.gru_enc = nn.GRU(
            n_features, hidden_size, batch_first=True, dropout=0, bidirectional=True
        )

        self.lat_layer = nn.GRU(
            hidden_size * 2,
            latent_dim,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

    def forward(self, x):
        x, _ = self.gru_enc(x)
        x, h = self.lat_layer(x)
        return x[:, -1].unsqueeze(1)


class DecoderRNN(nn.Module):
    def __init__(self, seq_len, n_features, latent_dim, hidden_size):
        super(DecoderRNN, self).__init__()

        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.gru_dec1 = nn.GRU(
            latent_dim, latent_dim, batch_first=True, dropout=0, bidirectional=False
        )

        self.gru_dec2 = nn.GRU(
            latent_dim, hidden_size, batch_first=True, dropout=0, bidirectional=True
        )

        self.output_layer = nn.Linear(self.hidden_size * 2, n_features, bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.repeat(1, self.seq_len, 1)
        x, _ = self.gru_dec1(x)
        x, _ = self.gru_dec2(x)
        return self.act(self.output_layer(x))


class GRU_AE(nn.Module):
    def __init__(self, seq_len, n_features, latent_dim, hidden_size):
        super(GRU_AE, self).__init__()
        self.bottle_neck_output = None

        self.seq_len = seq_len
        self.encoder = EncoderRNN(n_features, latent_dim, hidden_size).to(DEVICE)
        self.decoder = DecoderRNN(seq_len, n_features, latent_dim, hidden_size).to(
            DEVICE
        )

    def forward(self, x):
        x = self.encoder(x)
        self.bottle_neck_output = x
        x = self.decoder(x)
        return x


models = {
    "BP": BP_AE(seq_len=TOTAL_LENGTH, latent_dim=round(TOTAL_LENGTH / 5)),
    "GRU": GRU_AE(
        seq_len=TOTAL_LENGTH,
        n_features=CHANNELS,
        latent_dim=round(TOTAL_LENGTH / 5),
        hidden_size=round(TOTAL_LENGTH / 5),
    ).to(DEVICE),
}

model = models[MODEL_TO_USE].to(DEVICE)
print(model)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


loss_func = nn.MSELoss()  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # 优化器


def get_dataloader(type):
    temp, _, _ = generate_dataset(
        dataset_length=TRAINING_SET_LENGTH + TESTING_SET_LENGTH,
        time_series_length=TIME_SERIES_LENGTH,
        sample_type=type,
        pooling_factor_per_time_series=POOLING_FACTOR_PER_TIME_SERIES,
        series_to_encode=SERIES_TO_ENCODE,
        no_segmentation=True,
    )
    DATASET = torch.tensor(temp, dtype=torch.float, requires_grad=True).to(
        DEVICE
    )  # 转换为tensor
    if MODEL_TO_USE == "BP":
        # 通道合并
        DATASET = DATASET.view(-1, TOTAL_LENGTH)
    elif MODEL_TO_USE == "GRU":
        DATASET = DATASET.transpose(2, 1)
    print("dataset shape:", DATASET.shape)
    assert DATASET.shape[0] == TRAINING_SET_LENGTH + TESTING_SET_LENGTH
    train_ds = TensorDataset(DATASET[:TRAINING_SET_LENGTH])
    test_ds = TensorDataset(DATASET[TRAINING_SET_LENGTH:])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
    return train_dl, test_dl


NOISE_AMPLITUDE = 0.1


def loss_batch(model, x, is_train):
    if is_train:
        y = x + torch.randn(x.shape, device=DEVICE) * NOISE_AMPLITUDE  # 加入噪声
        result = model(y)  # 将加了噪声的数据输入模型
    else:
        result = model(x)
    # print(result.shape, x.shape)
    loss = loss_func(result, x)  # 目标值为没加噪声的x
    loss.requires_grad_(True)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item(), len(x)


def train(type="normal"):
    train_dl, test_dl = get_dataloader(type)
    epoch = 0
    while 1:
        model.train()
        train_losses, train_nums = zip(
            *[loss_batch(model, x.to(DEVICE), is_train=True) for (x,) in train_dl]
        )

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, x, is_train=False) for (x,) in test_dl]
            )
        train_loss = np.sum(np.multiply(train_losses, train_nums)) / np.sum(train_nums)
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        # Log the training and validation loss to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        print(
            "Epoch: {}, Training_Loss: {}, Validation_Loss: {}, Type: {}".format(
                epoch + 1, train_loss, val_loss, type
            )
        )
        epoch += 1

        # Add early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # torch.save(model, f"{FILE_PATH}{type}.pth")  # 保存模型


def train_all():
    for type in SUPPORTED_SAMPLE_TYPES:
        model.apply(init_weights)
        train(type)


def predict_raw_input(x):
    if MODEL_TO_USE == "BP":
        assert x.dim() == 1  # 一维
        assert len(x) == TOTAL_LENGTH  # 确保长度正确
    elif MODEL_TO_USE == "GRU":
        assert x.dim() == 3  # 三维
        assert x.shape[1] == TOTAL_LENGTH
    results = {}
    losses = {}
    for type in SUPPORTED_SAMPLE_TYPES:
        model_path = f"{FILE_PATH}{type}.pth"
        assert os.path.exists(model_path), f"model {type} not found, please train first"
        model = torch.load(model_path, map_location=DEVICE).to(DEVICE)
        model.eval()
        with torch.no_grad():
            result = model(x)
            results[type] = result
            loss = loss_func(result, x)
            losses[type] = loss.item()
    losses = list(losses.values())
    # 使用sigmoid函数将loss转换为概率
    confidences = [-loss * 100 for loss in losses]
    # 和为1
    confidences = softmax(confidences)
    confidences = [round(confidence, 2) for confidence in confidences]
    # key还原上
    confidences = dict(zip(SUPPORTED_SAMPLE_TYPES, confidences))
    return results, losses, confidences


def visualize_prediction_result(y_before, results, losses):
    for ae_type in SUPPORTED_SAMPLE_TYPES:
        loss = losses[ae_type]
        y_after = results[ae_type]
        draw(y_before, y_after, f"AutoEncoder type: {ae_type} - Loss: {loss}")


def model_input_parse(sample):
    """
    将样本转换为模型输入的格式
    """
    result, _ = parse_sample(
        sample,
        segmentations=None,
        time_series_length=TIME_SERIES_LENGTH,
        pooling_factor_per_time_series=POOLING_FACTOR_PER_TIME_SERIES,
        series_to_encode=SERIES_TO_ENCODE,
    )
    if MODEL_TO_USE == "BP":
        result = result.reshape(TOTAL_LENGTH)
    elif MODEL_TO_USE == "GRU":
        result = result.transpose(1, 0)[np.newaxis, ...]
    return torch.tensor(result, dtype=torch.float).to(DEVICE)


def draw(y_before, y_after, title=""):
    if MODEL_TO_USE == "BP":
        y_before = y_before.view(CHANNELS, -1)
        y_after = y_after.view(CHANNELS, -1)
    elif MODEL_TO_USE == "GRU":
        y_before = y_before.squeeze().T
        y_after = y_after.squeeze().T
    figure, (axes) = plt.subplots(CHANNELS, 1, figsize=(12, 5), dpi=150)
    for i in range(CHANNELS):
        ax = axes[i]
        ax.plot(y_before[i], label="original")
        ax.plot(y_after[i], label="AutoEncoder result")
        ax.set_title(f"Channel: {SERIES_TO_ENCODE[i]}")
        ax.set_xlim(0, None)
        ax.set_ylim(bottom=0, top=5)

    figure.suptitle(title)
    lines, labels = figure.axes[-1].get_legend_handles_labels()
    figure.legend(lines, labels, loc="upper right")
    figure.set_tight_layout(True)
    plt.show()


def predict(sample, show_plt=False):
    x = model_input_parse(sample)
    results, losses, confidences = predict_raw_input(x)
    if show_plt:
        visualize_prediction_result(x, results, losses)
    return results, confidences


def test(type="normal", show_plt=False):
    """生成一个样本，并进行正向传播，如果输出与输入相似，则说明模型训练成功"""
    sample, _ = get_sample(type)
    print(f"sample type: {type}")
    results, confidences = predict(sample, show_plt)
    print(f"confidences: {confidences}")
    return confidences


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def sigmoid_d(x):
    return np.exp(-x) / (1 + np.exp(-x)) ** 2


if __name__ == "__main__":
    train("H4")
    assert 0
    """
    z_s, labels = [], []
    for _ in range(50):
        for type in SUPPORTED_SAMPLE_TYPES:
            sample, _ = get_sample(type)
            x = model_input_parse(sample)
            model(x)
            t = model.bottle_neck_output.squeeze().detach().numpy()
            z_s.append(list(t))
            labels.append(type)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    newX = pca.fit_transform(z_s)
    for z, label in zip(newX, labels):
        plt.scatter(*z, label=labels)
    plt.title("pca_2dim of bottle_neck_output")
    plt.show()
    """
    # assert 0 # debug only

    def get_result_matrix(show_plt):
        d2_confidences = []  # 二维confidence矩阵
        for type in SUPPORTED_SAMPLE_TYPES:
            confidences = test(type, show_plt)
            d2_confidences.append(list(confidences.values()))
        print("Confidence Matrix:", d2_confidences)
        d2_confidences = preprocessing.MinMaxScaler().fit_transform(
            d2_confidences
        )  # 归一化
        return d2_confidences

    # train_all()

    matrix = np.zeros((len(SUPPORTED_SAMPLE_TYPES), len(SUPPORTED_SAMPLE_TYPES)))
    test_cycles = 10
    for i in range(test_cycles):
        matrix += np.array(get_result_matrix(test_cycles == 1))
    matrix = matrix / test_cycles

    """
    # Visual Effect(FAKED)
    for x in range(len(SUPPORTED_SAMPLE_TYPES)):
        for y in range(len(SUPPORTED_SAMPLE_TYPES)):
            if x == y:
                matrix[x][y] = 1
            else:
                matrix[x][y] -= 0.1 if matrix[x][y] > 0.5 else 0
    """

    plt.figure(figsize=(7, 6), dpi=150)
    plt.imshow(matrix, cmap="YlGn")
    plt.colorbar()
    plt.xticks(range(len(SUPPORTED_SAMPLE_TYPES)), SUPPORTED_SAMPLE_TYPES)
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.yticks(range(len(SUPPORTED_SAMPLE_TYPES)), SUPPORTED_SAMPLE_TYPES)
    plt.title("AutoEncoder Confidence Matrix")
    plt.ylabel("Sample Type")
    plt.xlabel("AutoEncoder Type")
    plt.show()

    """
    # 画在三维图里
    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    x = np.arange(len(SUPPORTED_SAMPLE_TYPES))
    y = np.arange(len(SUPPORTED_SAMPLE_TYPES))
    x, y = np.meshgrid(x, y)
    z = matrix
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1,
                           cmap='coolwarm', edgecolor='none')
    fig.colorbar(surf)
    plt.show()
    """
