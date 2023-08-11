"""
将bp、transformer、ae 三个分类器的结果进行融合
采用Fuzzy神经网络，具有可解释性

相关链接：
人工智能新进展，可解释性深度学习？深度卷积模糊系统理论及应用 - https://www.bilibili.com/read/cv11188724
神经网络可解释性、深度学习新方法 - https://www.bilibili.com/read/cv4311278
"""
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
FORCE_CPU = True  # 强制使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
print("Using device:", DEVICE)
EPOCHS = 500  # 训练数据集的轮次
LEARNING_RATE = 1e-3  # 学习率
N_CLASSES = len(SUPPORTED_SAMPLE_TYPES)
INPUT_VECTOR_SIZE = 3 * N_CLASSES  # 输入向量大小


class FuzzyLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FuzzyLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        fuzzy_degree_weights = torch.Tensor(self.input_dim, self.output_dim)
        self.fuzzy_degree = nn.Parameter(fuzzy_degree_weights)
        sigma_weights = torch.Tensor(self.input_dim, self.output_dim)
        self.sigma = nn.Parameter(sigma_weights)

        # initialize fuzzy degree and sigma parameters
        nn.init.xavier_uniform_(self.fuzzy_degree)  # fuzzy degree init
        nn.init.ones_(self.sigma)  # sigma init

    def forward(self, input):
        fuzzy_out = []
        for variable in input:
            fuzzy_out_i = torch.exp(
                -torch.sum(
                    torch.sqrt((variable - self.fuzzy_degree) / (self.sigma**2))
                )
            )
            if torch.isnan(fuzzy_out_i):
                fuzzy_out.append(variable)
            else:
                fuzzy_out.append(fuzzy_out_i)
        return torch.tensor(fuzzy_out, dtype=torch.float)


class FusedFuzzyDeepNet(nn.Module):
    def __init__(
        self,
        input_vector_size,
        fuzz_vector_size,
        num_class,
        fuzzy_layer_input_dim=1,
        fuzzy_layer_output_dim=1,
        dropout_rate=0.2,
        device=DEVICE,
    ):
        super(FusedFuzzyDeepNet, self).__init__()
        self.device = device
        self.input_vector_size = input_vector_size
        self.fuzz_vector_size = fuzz_vector_size
        self.num_class = num_class
        self.fuzzy_layer_input_dim = fuzzy_layer_input_dim
        self.fuzzy_layer_output_dim = fuzzy_layer_output_dim

        self.dropout_rate = dropout_rate

        self.bn = nn.BatchNorm1d(self.input_vector_size)
        self.fuzz_init_linear_layer = nn.Linear(
            self.input_vector_size, self.fuzz_vector_size
        )

        fuzzy_rule_layers = []
        for i in range(self.fuzz_vector_size):
            fuzzy_rule_layers.append(
                FuzzyLayer(fuzzy_layer_input_dim, fuzzy_layer_output_dim)
            )
        self.fuzzy_rule_layers = nn.ModuleList(fuzzy_rule_layers)

        self.dl_linear_1 = nn.Linear(self.input_vector_size, self.input_vector_size)
        self.dl_linear_2 = nn.Linear(self.input_vector_size, self.input_vector_size)
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.fusion_layer = nn.Linear(
            self.input_vector_size * 2, self.input_vector_size
        )
        self.output_layer = nn.Linear(self.input_vector_size, self.num_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        input = self.bn(input)
        fuzz_input = self.fuzz_init_linear_layer(input)
        fuzz_output = torch.zeros(input.size(), dtype=torch.float, device=self.device)
        for col_idx in range(fuzz_input.size()[1]):
            col_vector = fuzz_input[:, col_idx : col_idx + 1]
            fuzz_col_vector = (
                self.fuzzy_rule_layers[col_idx](col_vector).unsqueeze(0).view(-1, 1)
            )
            fuzz_output[:, col_idx : col_idx + 1] = fuzz_col_vector

        dl_layer_1_output = torch.sigmoid(self.dl_linear_1(input))
        dl_layer_2_output = torch.sigmoid(self.dl_linear_2(dl_layer_1_output))
        dl_layer_2_output = self.dropout_layer(dl_layer_2_output)

        cat_fuzz_dl_output = torch.cat([fuzz_output, dl_layer_2_output], dim=1)

        fused_output = torch.sigmoid(self.fusion_layer(cat_fuzz_dl_output))
        fused_output = torch.relu(fused_output)

        output = self.softmax(self.output_layer(fused_output))

        return output


model = FusedFuzzyDeepNet(
    input_vector_size=INPUT_VECTOR_SIZE, fuzz_vector_size=32, num_class=N_CLASSES
).to(
    DEVICE
)  # FNN模型
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
