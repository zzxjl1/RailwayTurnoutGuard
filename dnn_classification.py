"""
pytorch implementation of dnn classification
用dnn对提取的特征进行分类
至此论文复现完毕
"""

import os
import pickle
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from extract_features import calc_features
from alive_progress import alive_bar, alive_it
from sensor import SUPPORTED_SAMPLE_TYPES, generate_sample

FILENAME = "./models/dnn_classification.pth"
BATCH_SIZE = 64  # 每批处理的数据
FORCE_CPU = True  # 强制使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() and not FORCE_CPU
                      else 'cpu')
print('Using device:', DEVICE)
EPOCHS = 100  # 训练数据集的轮次
LEARNING_RATE = 1e-3  # 学习率

INPUT_VECTOR_SIZE = 1 + 10 * 3 * 4


def weight_init(m):
    """初始化权重"""
    if isinstance(m, nn.Conv3d):
        n = m.kernel_size[0] * m.kernel_size[1] * \
            m.kernel_size[2] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()


class BP_Net(nn.Module):
    def __init__(self, input_vector_size, output_vector_size):
        super(BP_Net, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_vector_size)
        self.fc1 = nn.Linear(input_vector_size, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, output_vector_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        bn1_result = self.bn1(x)

        fc1_result = self.fc1(bn1_result)
        bn2_result = self.bn2(fc1_result)
        x = F.relu(bn2_result)

        fc2_result = self.fc2(x)
        bn3_result = self.bn3(fc2_result)
        x = F.relu(bn3_result)

        out = self.out(x)
        softmax_result = self.softmax(out)
        return softmax_result


"""
# 定义BP模型结构
BP_Net = nn.Sequential(
    nn.BatchNorm1d(INPUT_VECTOR_SIZE),  # 归一化
    nn.Linear(INPUT_VECTOR_SIZE, 64),  # 全连接层
    nn.BatchNorm1d(64),
    nn.ReLU(),  # 激活函数
    nn.Linear(64, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128, 12),
    nn.Softmax(dim=1)  # 分类任务最后用softmax层
)
"""


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
            fuzzy_out_i = torch.exp(-torch.sum(torch.sqrt(
                (variable - self.fuzzy_degree) / (self.sigma ** 2))))
            if torch.isnan(fuzzy_out_i):
                fuzzy_out.append(variable)
            else:
                fuzzy_out.append(fuzzy_out_i)
        return torch.tensor(fuzzy_out, dtype=torch.float)


class FusedFuzzyDeepNet(nn.Module):
    def __init__(self, input_vector_size, fuzz_vector_size, num_class, fuzzy_layer_input_dim=1,
                 fuzzy_layer_output_dim=1,
                 dropout_rate=0.5, device=DEVICE):

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
            self.input_vector_size, self.fuzz_vector_size)

        fuzzy_rule_layers = []
        for i in range(self.fuzz_vector_size):
            fuzzy_rule_layers.append(FuzzyLayer(
                fuzzy_layer_input_dim, fuzzy_layer_output_dim))
        self.fuzzy_rule_layers = nn.ModuleList(fuzzy_rule_layers)

        self.dl_linear_1 = nn.Linear(
            self.input_vector_size, self.input_vector_size)
        self.dl_linear_2 = nn.Linear(
            self.input_vector_size, self.input_vector_size)
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.fusion_layer = nn.Linear(
            self.input_vector_size * 2, self.input_vector_size)
        self.output_layer = nn.Linear(self.input_vector_size, self.num_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        input = self.bn(input)
        fuzz_input = self.fuzz_init_linear_layer(input)
        fuzz_output = torch.zeros(
            input.size(), dtype=torch.float, device=self.device)
        for col_idx in range(fuzz_input.size()[1]):
            col_vector = fuzz_input[:, col_idx:col_idx + 1]
            fuzz_col_vector = self.fuzzy_rule_layers[col_idx](
                col_vector).unsqueeze(0).view(-1, 1)
            fuzz_output[:, col_idx:col_idx + 1] = fuzz_col_vector

        dl_layer_1_output = torch.sigmoid(self.dl_linear_1(input))
        dl_layer_2_output = torch.sigmoid(self.dl_linear_2(dl_layer_1_output))
        dl_layer_2_output = self.dropout_layer(dl_layer_2_output)

        cat_fuzz_dl_output = torch.cat([fuzz_output, dl_layer_2_output], dim=1)

        fused_output = torch.sigmoid(self.fusion_layer(cat_fuzz_dl_output))
        fused_output = torch.relu(fused_output)

        output = self.softmax(self.output_layer(fused_output))

        return output


model = FusedFuzzyDeepNet(input_vector_size=INPUT_VECTOR_SIZE,
                          fuzz_vector_size=30, num_class=12).to(DEVICE)  # 使用FNN模型
# model = BP_Net(input_vector_size=INPUT_VECTOR_SIZE,
#               output_vector_size=12).to(DEVICE)  # 使用BP模型
print(model)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # adam优化器
loss_func = nn.CrossEntropyLoss()  # 交叉熵损失函数
model.apply(weight_init)  # 预初始化权重


def get_data(train_ds, valid_ds):  # 获取dataloader
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True),
    )


def contains_nan(outputs):
    """判断是否有nan,nan会导致训练失败"""
    outputs = np.array(outputs)
    t = int((outputs != outputs).sum())
    return t > 0


def loss_batch(model, loss_func, xb, yb):
    optimizer.zero_grad()
    output = model(xb)  # 前向传播
    """
    print("input", xb)
    print("output", output)
    print("target", yb)
    """
    loss = loss_func(output, yb)  # 计算损失
    loss.requires_grad_(True)
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    return loss.item(), len(xb)


def fit(train_dl, valid_dl):
    for step in range(EPOCHS):  # 训练轮次
        model.train()  # 训练模式
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb)

        model.eval()  # 验证模式
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print('当前step:' + str(step), '验证集损失：' + str(val_loss))


def generate_data(num):
    """生成数据集（带缓存）"""
    filename = 'extracted_features_dataset.pkl'
    if os.path.exists(filename):  # 如果缓存存在
        with open(filename, 'rb') as f:
            x, y = pickle.load(f)  # 读取
            if len(x) >= num:  # 如果数量符合要求
                # 返回前num个
                return x[:num], y[:num]
            else:  # 否则重新生成
                print('dataset cache not match, regenerate')
                os.remove(filename)
    x = []  # 特征值
    y = []  # 目标值
    with alive_bar(num, title="数据集生成中") as bar:
        while len(x) < num:  # 生成num个数据
            for sample_type in SUPPORTED_SAMPLE_TYPES:  # 每个类型
                if len(x) >= num:  # 如果数量符合要求
                    break
                sample, _ = generate_sample(sample_type)  # 生成样本
                t = list(calc_features(sample).values())  # 计算特征
                # STANDARDIZE
                #t = (t - np.mean(t)) / np.std(t)
                assert not contains_nan(t)  # 检查是否有nan
                x.append(t)
                # one-hot encoding
                index = SUPPORTED_SAMPLE_TYPES.index(sample_type)
                y.append([0] * index + [1] + [0] *
                         (len(SUPPORTED_SAMPLE_TYPES) - index - 1))
                bar()  # 进度条+1
    with open(filename, 'wb') as f:
        pickle.dump((x, y), f)  # 保存到磁盘
    return x, y


def predict_raw_input(x):
    """预测,输入为原始数据，直接入模型"""
    assert os.path.exists(
        FILENAME), "model not found, please run train() first!"
    model = torch.load(FILENAME)  # 加载模型
    model.eval()  # 验证模式
    with torch.no_grad():
        output = model(x)
        return output


def train():
    """训练模型"""
    DATASET_LENGTH = 1000  # 数据集长度
    TRANING_SET_LENGTH = 800  # 训练集长度
    DATASET = generate_data(DATASET_LENGTH)  # 生成数据集

    x, y = map(lambda a: torch.tensor(np.array(a), dtype=torch.float,
               requires_grad=True), DATASET)  # 转换为tensor
    x, y = x.to(DEVICE), y.to(DEVICE)
    print(x.shape, y.shape)
    #print(x[0], y[0])
    train_ds = TensorDataset(x[:TRANING_SET_LENGTH],
                             y[:TRANING_SET_LENGTH])  # 训练集
    valid_ds = TensorDataset(
        x[TRANING_SET_LENGTH:], y[TRANING_SET_LENGTH:])  # 验证集
    train_dl, valid_dl = get_data(train_ds, valid_ds)  # 转换为dataloader
    fit(train_dl, valid_dl)  # 开始训练

    torch.save(model, FILENAME)  # 保存模型
    # torch.onnx.export(model, torch.randn(1, INPUT_VECTOR_SIZE),"model.onnx")  # 保存onnx格式模型


def parse_predict_result(result):
    """解析预测结果"""
    result_pretty = [round(i, 2) for i in result.tolist()[0]]
    result_pretty = dict(zip(SUPPORTED_SAMPLE_TYPES, result_pretty))  # 让输出更美观
    return result_pretty


def get_label_from_result_pretty(result_pretty):
    """从解析后的预测结果中获取标签"""
    return max(result_pretty, key=result_pretty.get)


def predict(sample):
    """预测"""
    features = list(calc_features(sample).values())  # 计算特征
    features = torch.tensor([features], dtype=torch.float,
                            requires_grad=True)  # 转换为tensor
    result = predict_raw_input(features.to(DEVICE))  # 预测
    result_pretty = parse_predict_result(result)  # 解析结果
    return result_pretty


def test(type="normal"):
    """生成type类型的样本，然后跑模型预测，最后返回是否正确"""
    sample, _ = generate_sample(type)  # 生成样本
    result = predict(sample)  # 预测
    print(result)
    label = get_label_from_result_pretty(result)  # 获取预测结果标签字符串
    print(label)
    return label == type  # 预测是否正确


if __name__ == '__main__':

    train()  # 训练模型，第一次运行时需要先训练模型，训练完会持久化权重至硬盘请注释掉这行

    test_cycles = 100  # 测试次数
    test_results = []
    for _ in range(test_cycles):
        t = test(random.choice(SUPPORTED_SAMPLE_TYPES))  # 随机生成一个类型的样本，然后预测
        test_results.append(t)  # 记录结果
    print("accuracy:", test_results.count(
        True) / test_cycles)  # 输出正确率（94.5%左右）
