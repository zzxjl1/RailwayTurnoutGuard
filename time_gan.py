import os
from matplotlib import pyplot as plt
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn

from sensor import SAMPLE_RATE, SUPPORTED_SAMPLE_TYPES
from sensor.dataset import generate_dataset


class GRUNet(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, n_layers, max_seq_len, use_activation, device
    ):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True).to(
            self.device
        )
        self.fc = nn.Linear(hidden_dim, output_dim).to(self.device)
        self.activation = nn.Sigmoid()
        self.max_seq_len = max_seq_len
        self.padding_value = -1.0  # from PyTorch implementation
        self.use_activation = use_activation

        with torch.no_grad():
            for name, param in self.gru.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "bias_ih" in name:
                    param.data.fill_(1)
                elif "bias_hh" in name:
                    param.data.fill_(0)
            for name, param in self.fc.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    param.data.fill_(0)

    def forward(self, X, T):
        X_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=X, lengths=T, batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(X_packed)
        out, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=out,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len,
        )
        out = self.fc(out)
        if self.use_activation:
            out = self.activation(out)
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden


class RecurrentNetwork:
    def __init__(
        self,
        module_name,
        num_layers,
        input_dimension,
        hidden_dimension,
        output_dimension,
        max_seq_length,
        device,
        use_activation=True,
    ) -> None:
        self.module_name = module_name
        assert self.module_name in ["gru", "lstm"]  # lstmLN is missing
        self.num_layers = num_layers
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        self.max_seq_length = max_seq_length
        self.device = device
        self.use_activation = use_activation
        self.network = GRUNet(
            self.input_dimension,
            self.hidden_dimension,
            self.output_dimension,
            self.num_layers,
            self.max_seq_length,
            self.use_activation,
            self.device
        )

    def __call__(self, X, T):
        out = self.network.forward(X, T)
        return out


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
      - data: original data

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))

    return time, max_seq_len


def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    """Random vector generation.

    Args:
      - batch_size: size of the random vector
      - z_dim: dimension of random vector
      - T_mb: time information for the random vector
      - max_seq_len: maximum sequence length

    Returns:
      - Z_mb: generated random vector
    """
    Z_mb = list()
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim])
        temp_Z = np.random.uniform(0.0, 1, [T_mb[i], z_dim])
        temp[: T_mb[i], :] = temp_Z
        Z_mb.append(temp_Z)
    return Z_mb


class TimeSeriesDataLoader:
    def __init__(self, sample_type, device) -> None:
        super().__init__()
        self.device = device
        assert sample_type in SUPPORTED_SAMPLE_TYPES
        self.data_dir = os.path.join(os.path.dirname(__file__), "../data")

        data, _, _ = generate_dataset(dataset_length=DATASET_LENGTH,
                                      time_series_length=TIME_SERIES_LENGTH,
                                      sample_type=sample_type,
                                      pooling_factor_per_time_series=POOLING_FACTOR_PER_TIME_SERIES,
                                      series_to_encode=SERIES_TO_ENCODE)
        data = data.transpose(0, 2, 1)

        self.data, self.min_val, self.max_val = self.MinMaxScaler(
            np.asarray(data))
        self.num_obs, self.seq_len, self.dim = self.data.shape
        self.get_time()

        print("dataset shape:", self.data.shape)

    def MinMaxScaler(self, data):
        """
        Min-Max Normalizer.
        Different from the one in the utils.py

        Args:
            - data: raw data

        Returns:
            - norm_data: normalized data
            - min_val: minimum values (for renormalization)
            - max_val: maximum values (for renormalization)
        """
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val

        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)

        return norm_data, min_val, max_val

    def get_time(self):
        self.T, self.max_seq_len = extract_time(self.data)

    def get_z(self, batch_size, T_mb):
        if not isinstance(T_mb, list):
            T_mb = list(T_mb.numpy())
        T_mb = list(map(int, T_mb))
        return torch.from_numpy(
            np.asarray(
                random_generator(batch_size, self.dim, T_mb, self.max_seq_len),
                dtype=np.float32,
            )
        ).to(self.device)

    def get_x_t(self, batch_size):
        idx = [i for i in range(self.num_obs)]
        random.shuffle(idx)
        idx = idx[:batch_size]
        batch_data = np.take(
            np.array(self.data, dtype=np.float32), idx, axis=0)
        batch_data_T = np.take(np.array(self.T, dtype=np.float32), idx, axis=0)
        return (
            torch.from_numpy(batch_data).to(self.device),
            torch.from_numpy(
                batch_data_T
            ),  # T should a simple CPU tensor, otherwise error is thrown
        )

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.data[index, :]),
            torch.from_numpy(self.T[index]),
            torch.from_numpy(self.Z[index, :]),
        )

    def __len__(self):
        return self.num_obs


class TimeGAN:
    def __init__(self, parameters):

        self.parameters = parameters
        self.device = self.parameters["device"]
        self.use_wgan = self.parameters["use_wgan"]

        # Network Parameters
        self.hidden_dim = self.parameters["hidden_dim"]
        self.num_layers = self.parameters["num_layer"]
        self.iterations = self.parameters["iterations"]
        self.batch_size = self.parameters["batch_size"]
        self.sequence_length = self.parameters["sequence_length"]
        self.sample_type = self.parameters["sample_type"]
        self.dataloader = TimeSeriesDataLoader(
            self.sample_type, self.device) if self.sample_type else None

        if self.dataloader:
            self.max_seq_length = self.dataloader.max_seq_len
            self.ip_dimension = self.dataloader.dim
            self.data_min_val = self.dataloader.min_val
            self.data_max_val = self.dataloader.max_val
            self.ori_time = self.dataloader.T
        else:
            self.max_seq_length = self.parameters["sequence_length"]
            self.ip_dimension = self.parameters["channels"]
            self.data_min_val = 0
            self.data_max_val_range = self.parameters["max_val_range"]

        self.gamma = 1
        self.c = 0.01  # clipping value
        self.initialize_networks()
        self.initialize_optimizers()
        self.loss_bce = torch.nn.functional.binary_cross_entropy_with_logits
        self.loss_mse = torch.nn.functional.mse_loss
        self.file_storage()

    def file_storage(self):
        save_dir = os.path.join(os.path.dirname(
            __file__), ("./models/time_gan"))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.model_save_path = os.path.join(
            save_dir, f"{self.sample_type}.pth")

    def save_model(self):
        print("Saving model...")
        torch.save(
            {
                "embedder_state_dict": self.embedder.network.state_dict(),
                "recovery_state_dict": self.recovery.network.state_dict(),
                "supervisor_state_dict": self.supervisor.network.state_dict(),
                "generator_state_dict": self.generator.network.state_dict(),
                "discriminator_state_dict": self.discriminator.network.state_dict(),
            },
            self.model_save_path,
        )
        print("Saving model complete!!!")

    def load_model(self, path):
        check_point = torch.load(path, map_location=DEVICE)
        self.embedder.network.load_state_dict(
            check_point["embedder_state_dict"])
        self.recovery.network.load_state_dict(
            check_point["recovery_state_dict"])
        self.supervisor.network.load_state_dict(
            check_point["supervisor_state_dict"])
        self.generator.network.load_state_dict(
            check_point["generator_state_dict"])
        self.discriminator.network.load_state_dict(
            check_point["discriminator_state_dict"]
        )

    def initialize_networks(self):
        self.embedder = RecurrentNetwork(
            "gru",
            self.num_layers,
            self.ip_dimension,
            self.hidden_dim,
            self.hidden_dim,
            self.max_seq_length,
            self.device
        )
        self.recovery = RecurrentNetwork(
            "gru",
            self.num_layers,
            self.hidden_dim,
            self.hidden_dim,
            self.ip_dimension,
            self.max_seq_length,
            self.device,
            use_activation=False,
        )

        self.discriminator = RecurrentNetwork(
            "gru",
            self.num_layers,
            self.hidden_dim,
            self.hidden_dim,
            1,
            self.max_seq_length,
            self.device,
            use_activation=False,
        )
        self.generator = RecurrentNetwork(
            "gru",
            self.num_layers,
            self.ip_dimension,
            self.hidden_dim,
            self.hidden_dim,
            self.max_seq_length,
            self.device
        )

        self.supervisor = RecurrentNetwork(
            "gru",
            self.num_layers - 1,
            self.hidden_dim,
            self.hidden_dim,
            self.hidden_dim,
            self.max_seq_length,
            self.device
        )

    def initialize_optimizers(self):
        self.E0_solver_params = [
            *self.embedder.network.parameters(),
            *self.recovery.network.parameters(),
        ]
        self.E0_solver = torch.optim.Adam(
            self.E0_solver_params, lr=self.parameters["learning_rate"]
        )

        self.E_solver_params = [
            *self.embedder.network.parameters(),
            *self.recovery.network.parameters(),
        ]
        self.E_solver = torch.optim.Adam(
            self.E_solver_params, lr=self.parameters["learning_rate"]
        )

        self.D_solver_params = [
            *self.discriminator.network.parameters(),
        ]
        self.D_solver = torch.optim.Adam(
            self.D_solver_params, lr=self.parameters["learning_rate"]
        )

        self.G_solver_params = [
            *self.generator.network.parameters(),
            *self.supervisor.network.parameters(),
        ]
        self.G_solver = torch.optim.Adam(
            self.G_solver_params, lr=self.parameters["learning_rate"]
        )

        self.GS_solver_params = [
            *self.generator.network.parameters(),
            *self.supervisor.network.parameters(),
        ]
        self.GS_solver = torch.optim.Adam(
            self.GS_solver_params, lr=self.parameters["learning_rate"]
        )

    def embedder_recovery_training(self):
        print("Traning Embedder and Recovery Networks...")
        for i in tqdm(range(self.iterations)):
            self.E0_solver.zero_grad()
            X, T = self.dataloader.get_x_t(self.batch_size)
            H = self.embedder(X, T)
            X_tilde = self.recovery(H, T)
            E_loss_0 = 10 * torch.sqrt(self.loss_mse(X, X_tilde))
            E_loss_0.backward()
            self.E0_solver.step()
            print("EPOCH:{} embedder_recovery_error:{}".format(
                i, str(E_loss_0.item())))
        print("Traning Embedder and Recovery Networks complete")

    def supervisor_training(self):
        print("Traning Supervisor Network...")
        for i in tqdm(range(self.iterations)):
            self.GS_solver.zero_grad()
            X, T = self.dataloader.get_x_t(self.batch_size)
            # Z = self.dataloader.get_z(self.batch_size) # no use, same is the case in the main implementation
            H = self.embedder(X, T)
            H_hat_supervise = self.supervisor(H, T)
            G_loss_S = self.loss_mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])
            G_loss_S.backward()
            self.GS_solver.step()

            print("EPOCH:{} supervisor_error:{}".format(i, str(G_loss_S.item())))
        print("Traning Supervisor Network complete")

    def joint_training(self):
        print("Performing Joint Network training...")
        for i in tqdm(range(self.iterations)):
            for kk in range(2):
                X, T = self.dataloader.get_x_t(self.batch_size)
                Z = self.dataloader.get_z(self.batch_size, T)

                self.G_solver.zero_grad()
                H = self.embedder(X, T)
                H_hat_supervise = self.supervisor(H, T)

                E_hat = self.generator(Z, T)
                H_hat = self.supervisor(E_hat, T)
                X_hat = self.recovery(H_hat, T)

                Y_fake = self.discriminator(E_hat, T)
                Y_fake_e = self.discriminator(H_hat, T)

                G_loss_U = self.loss_bce(Y_fake, torch.ones_like(Y_fake))
                G_loss_U_e = self.loss_bce(Y_fake_e, torch.ones_like(Y_fake_e))
                G_loss_S = self.loss_mse(
                    H[:, 1:, :], H_hat_supervise[:, :-1, :])

                # Two Momments
                G_loss_V1 = torch.mean(
                    torch.abs(
                        torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6)
                        - torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)
                    )
                )
                G_loss_V2 = torch.mean(
                    torch.abs((X_hat.mean(dim=0)) - (X.mean(dim=0))))
                G_loss_V = G_loss_V1 + G_loss_V2

                G_loss = (
                    G_loss_U
                    + self.gamma * G_loss_U_e
                    + 100 * torch.sqrt(G_loss_S)
                    + 100 * G_loss_V
                )
                G_loss.backward()
                self.G_solver.step()

                self.E_solver.zero_grad()
                H = self.embedder(X, T)
                H_hat_supervise = self.supervisor(H, T)
                X_tilde = self.recovery(H, T)

                G_loss_S = self.loss_mse(
                    H[:, 1:, :], H_hat_supervise[:, :-1, :])
                E_loss_0 = 10 * torch.sqrt(self.loss_mse(X, X_tilde))

                E_loss = E_loss_0 + 0.1 * G_loss_S
                E_loss.backward()
                self.E_solver.step()

                print("EPOCH:{} joint_generator_error {}".format(
                    i * 2 + kk, str(G_loss.item())))
                print("EPOCH:{} joint_embedder_recovery_error:{}".format(
                    i * 2 + kk, str(E_loss.item())))

            X, T = self.dataloader.get_x_t(self.batch_size)
            Z = self.dataloader.get_z(self.batch_size, T)

            self.D_solver.zero_grad()
            H = self.embedder(X, T)
            E_hat = self.generator(Z, T)
            H_hat = self.supervisor(E_hat, T)

            Y_real = self.discriminator(H, T)
            Y_fake = self.discriminator(E_hat, T)
            Y_fake_e = self.discriminator(H_hat, T)
            D_loss_real = self.loss_bce(Y_real, torch.ones_like(Y_real))
            D_loss_fake = self.loss_bce(Y_fake, torch.zeros_like(Y_fake))
            D_loss_fake_e = self.loss_bce(Y_fake_e, torch.zeros_like(Y_fake_e))

            D_loss = D_loss_real + D_loss_fake + self.gamma * D_loss_fake_e
            if D_loss > 0.15:
                D_loss.backward()
                self.D_solver.step()

            print("EPOCH:{} joint_discriminator_error:{}".format(
                i, str(D_loss.item())))
        print("Joint Network training complete")

    def joint_training_wgan(self):
        print("Performing Joint Network training...")
        for i in tqdm(range(self.iterations)):
            for kk in range(4):
                X, T = self.dataloader.get_x_t(self.batch_size)
                Z = self.dataloader.get_z(self.batch_size, T)

                self.G_solver.zero_grad()
                H = self.embedder(X, T)
                H_hat_supervise = self.supervisor(H, T)

                E_hat = self.generator(Z, T)
                H_hat = self.supervisor(E_hat, T)
                X_hat = self.recovery(H_hat, T)

                Y_fake = self.discriminator(E_hat, T)
                Y_fake_e = self.discriminator(H_hat, T)

                G_loss_U = -torch.mean(Y_fake)
                G_loss_U_e = -torch.mean(Y_fake_e)
                G_loss_S = self.loss_mse(
                    H[:, 1:, :], H_hat_supervise[:, :-1, :])

                # Two Momments
                G_loss_V1 = torch.mean(
                    torch.abs(
                        torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6)
                        - torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)
                    )
                )
                G_loss_V2 = torch.mean(
                    torch.abs((X_hat.mean(dim=0)) - (X.mean(dim=0))))
                G_loss_V = G_loss_V1 + G_loss_V2

                G_loss = (
                    G_loss_U
                    + self.gamma * G_loss_U_e
                    + 100 * torch.sqrt(G_loss_S)
                    + 100 * G_loss_V
                )
                G_loss.backward()
                self.G_solver.step()

                self.E_solver.zero_grad()
                H = self.embedder(X, T)
                H_hat_supervise = self.supervisor(H, T)
                X_tilde = self.recovery(H, T)

                G_loss_S = self.loss_mse(
                    H[:, 1:, :], H_hat_supervise[:, :-1, :])
                E_loss_0 = 10 * torch.sqrt(self.loss_mse(X, X_tilde))

                E_loss = E_loss_0 + 0.1 * G_loss_S
                E_loss.backward()
                self.E_solver.step()

                print("EPOCH:{} joint_generator_error:{}".format(
                    i * 2 + kk, str(G_loss.item())))
                print("EPOCH:{} joint_embedder_recovery_error:{}".format(
                    i * 2 + kk, str(E_loss.item())))

            X, T = self.dataloader.get_x_t(self.batch_size)
            Z = self.dataloader.get_z(self.batch_size, T)

            self.D_solver.zero_grad()
            H = self.embedder(X, T)
            E_hat = self.generator(Z, T)
            H_hat = self.supervisor(E_hat, T)

            Y_real = self.discriminator(H, T)
            Y_fake = self.discriminator(E_hat, T)
            Y_fake_e = self.discriminator(H_hat, T)
            D_loss_real = -torch.mean(Y_real)
            D_loss_fake = torch.mean(Y_fake)
            D_loss_fake_e = torch.mean(Y_fake_e)

            D_loss = D_loss_real + D_loss_fake + self.gamma * D_loss_fake_e
            D_loss.backward()
            self.D_solver.step()

            # clipping D
            for p in self.discriminator.network.parameters():
                p.data.clamp_(-self.c, self.c)

            print("EPOCH:{} joint_discriminator_error:{}".format(
                i, str(D_loss.item())))
        print("Joint Network training complete")

    def synthetic_data_generation(self, num, seq_len):
        T = [seq_len for _ in range(num)]
        T_mb = list(map(int, T))
        Z = torch.from_numpy(
            np.asarray(
                random_generator(num, self.ip_dimension,
                                 T_mb, self.max_seq_length),
                dtype=np.float32,
            )
        ).to(self.device)

        E_hat = self.generator(Z, T)
        H_hat = self.supervisor(E_hat, T)
        X_hat = self.recovery(H_hat, T)

        generated_data = list()
        # data generated has max-length, so to match the number of datapoints as in original data
        for i in range(num):
            temp = X_hat[i, : T[i], :]
            generated_data.append(temp)

        # Renormalization
        generated_data_scaled = []
        for x in generated_data:
            x = x.cpu().detach().numpy()
            x = (x * random.uniform(*self.data_max_val_range)) + self.data_min_val
            generated_data_scaled.append(x)
        return generated_data_scaled

    def train(self):
        self.joint_trainer_fn = (
            self.joint_training_wgan if self.use_wgan else self.joint_training
        )
        try:
            self.embedder_recovery_training()
            self.save_model()
            self.supervisor_training()
            self.save_model()
            self.joint_trainer_fn()
            self.save_model()
        except KeyboardInterrupt:
            print("KeyBoard Interrupt!")
            self.save_model()


DATASET_LENGTH = 5000  # 数据集总长度
TIME_SERIES_DURATION = 20  # 20s
TIME_SERIES_LENGTH = SAMPLE_RATE * TIME_SERIES_DURATION  # 采样率*时间，总共的数据点数
SERIES_TO_ENCODE = ['A', 'B', 'C']  # 生成三相电流序列，不生成power曲线
POOLING_FACTOR_PER_TIME_SERIES = 10  # 每个时间序列的池化因子,用于降低工作量

EPOCHS = 10000  # 训练数据集的轮次
LEARNING_RATE = 1e-3  # 学习率
BATCH_SIZE = 64  # 每批处理的数据
FORCE_CPU = True  # 强制使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      and not FORCE_CPU else "cpu")

model_params = {
    "device": DEVICE,
    "hidden_dim": 120,
    "num_layer": 3,
    "iterations": EPOCHS,
    "batch_size": BATCH_SIZE,
    "sequence_length": TIME_SERIES_LENGTH//POOLING_FACTOR_PER_TIME_SERIES,
    "learning_rate": LEARNING_RATE,
    "use_wgan": False,
    "sample_type": None,
    "channels": len(SERIES_TO_ENCODE)
}


def generate(type, num=1, show_plot=False):
    max_val_range_dict = {
        "F1": (0.3, 0.5),
        "F2": (0, 0.01)
    }
    model_params["max_val_range"] = max_val_range_dict[type] if type in max_val_range_dict else (
        4.5, 5.5)
    model = TimeGAN(model_params)
    file_path = f"./models/time_gan/{type}.pth"
    assert os.path.exists(file_path), "model not exists, please train() first!"
    model.load_model(file_path)
    fake_data = model.synthetic_data_generation(
        num, TIME_SERIES_LENGTH//POOLING_FACTOR_PER_TIME_SERIES)
    result = np.array(fake_data)
    result = result.transpose(0, 2, 1)
    # print(result.shape)

    if show_plot:
        for sample in result:  # 如果一次生成多个样本，则循环显示
            for i, channel in enumerate(sample):  # 多个通道
                plt.plot(channel, label=SERIES_TO_ENCODE[i])
            plt.title(f"TimeGAN Generation Result - Type: {type}")
            plt.legend(loc="upper right")
            plt.show()

    return result


def train(type):
    model_params["sample_type"] = type
    model = TimeGAN(model_params)
    model.train()


def train_all():
    for type in SUPPORTED_SAMPLE_TYPES:
        print(f"Training for type: {type}")
        train(type)


if __name__ == "__main__":

    # train_all()
    fig, _ = plt.subplots(3, 4, figsize=(10, 10), dpi=150)
    fig.subplots_adjust(hspace=0.5)
    for i, type in enumerate(SUPPORTED_SAMPLE_TYPES):
        sample = generate(type, num=1, show_plot=False)[0]
        plt.subplot(3, 4, i+1)
        # 坐标从0开始
        plt.xlim(0, TIME_SERIES_LENGTH//POOLING_FACTOR_PER_TIME_SERIES)
        plt.ylim(0, 5)
        plt.xticks([])
        plt.yticks([])
        plt.title(type, fontsize=10)
        lines = []
        labels = []
        for i, channel in enumerate(sample):  # 多个通道
            line = plt.plot(channel)
            lines.append(line)
            labels.append(SERIES_TO_ENCODE[i])
    fig.legend(lines, labels=labels, loc="upper right")
    plt.suptitle("TimeGAN Generation Result")
    plt.show()
