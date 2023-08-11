import os
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import random
import numpy as np
import torch
from torch import nn
from gru_score import GRUScore
from segmentation import calc_segmentation_points
from sensor.config import SUPPORTED_SAMPLE_TYPES
from sensor.simulate import generate_sample
from tool_utils import get_label_from_result_pretty, parse_predict_result
import auto_encoder
import mlp_classification
import gru_classification
from gru_classification import GRU_FCN, Vanilla_GRU, FCN_1D, Squeeze_Excite
from mlp_classification import MLP
from auto_encoder import BP_AE, EncoderRNN, DecoderRNN, GRU_AE
from alive_progress import alive_it

FILE_PATH = "./models/result_fusion.pth"
TRANING_SET_LENGTH = 400  # 训练集长度
TESTING_SET_LENGTH = 100  # 测试集长度
DATASET_LENGTH = TRANING_SET_LENGTH + TESTING_SET_LENGTH
BATCH_SIZE = 64  # 每批处理的数据
FORCE_CPU = False  # 强制使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
print("Using device:", DEVICE)
EPOCHS = 2000  # 训练数据集的轮次
LEARNING_RATE = 1e-4  # 学习率
N_CLASSES = len(SUPPORTED_SAMPLE_TYPES)
INPUT_VECTOR_SIZE = 3 * N_CLASSES  # 输入向量大小


model = nn.Sequential(
    nn.BatchNorm1d(INPUT_VECTOR_SIZE),  # 归一化
    nn.Linear(INPUT_VECTOR_SIZE, 64),  # 全连接
    nn.ReLU(),  # 激活函数
    nn.Linear(64, N_CLASSES),
    nn.Softmax(dim=1),  # 分类任务最后用softmax层
).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_func = nn.CrossEntropyLoss()


def model_input_parse(sample, segmentations, batch_simulation=True):
    if segmentations is None:
        segmentations = calc_segmentation_points(sample)
    bp_result = mlp_classification.predict(sample, segmentations).to(DEVICE)
    gru_result = gru_classification.predict(sample, segmentations).to(DEVICE)
    _, ae_result = auto_encoder.predict(sample)
    ae_result = torch.tensor(list(ae_result.values()), dtype=torch.float).to(DEVICE)

    print("bp_result:", bp_result)
    print("gru_result:", gru_result)
    print("ae_result:", ae_result)
    # 拼接三个分类器的结果
    result = torch.cat([bp_result, gru_result, ae_result], dim=0)
    if batch_simulation:
        result = result.unsqueeze(0)
    print(result.shape)
    return result


def predict(sample, segmentations=None):
    assert os.path.exists(FILE_PATH), "model file not exists, please train first"
    model = torch.load(FILE_PATH, map_location=DEVICE).to(DEVICE)
    model_input = model_input_parse(sample, segmentations)
    model.eval()
    with torch.no_grad():
        output = model(model_input)
    print(output)
    return output.squeeze()


def generate_dataset():
    x, y = [], []
    for i in alive_it(range(DATASET_LENGTH)):
        type = random.choice(SUPPORTED_SAMPLE_TYPES)
        sample, segmentations = generate_sample(type)
        model_input = model_input_parse(sample, segmentations, batch_simulation=False)
        x.append(model_input.detach().cpu().numpy())
        index = SUPPORTED_SAMPLE_TYPES.index(type)
        # one-hot编码
        y.append([0] * index + [1] + [0] * (len(SUPPORTED_SAMPLE_TYPES) - index - 1))
    x = torch.tensor(np.array(x), dtype=torch.float).to(DEVICE)
    y = torch.tensor(np.array(y), dtype=torch.float).to(DEVICE)
    print(x.shape, y.shape)
    train_ds = TensorDataset(x[:TRANING_SET_LENGTH], y[:TRANING_SET_LENGTH])  # 训练集
    valid_ds = TensorDataset(x[TRANING_SET_LENGTH:], y[TRANING_SET_LENGTH:])  # 验证集
    return train_ds, valid_ds


def train():
    for epoch in range(EPOCHS):
        for i, (x, y) in enumerate(train_dl):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            output = model(x)
            loss = loss_func(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch:", epoch, "batch:", i, "loss:", loss.item())

    torch.save(model, FILE_PATH)


def test():
    model = torch.load(FILE_PATH, map_location=DEVICE).to(DEVICE)  # 加载模型
    model.eval()  # 验证模式
    correct = 0
    total = 0
    for i, (x, y) in enumerate(valid_dl):
        y = y.float().to(DEVICE)
        output = model(x)
        _, predicted = torch.max(output.data, 1)
        _, label = torch.max(y.data, 1)
        total += y.size(0)
        correct += (predicted == label).sum().item()
    print("accu:", correct / total)


def get_dataloader(train_ds, valid_ds):
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True)
    return train_dl, valid_dl


if __name__ == "__main__":
    train_ds, valid_ds = generate_dataset()  # 生成数据集
    train_dl, valid_dl = get_dataloader(train_ds, valid_ds)  # 转换为dataloader
    train()  # 训练模型，第一次运行时需要先训练模型，训练完会持久化权重至硬盘请注释掉这行

    test()
