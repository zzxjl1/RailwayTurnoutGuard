"""
基于lstm的gan，更适用于序列数据
"""

import numpy as np
import torch
import torch.nn as nn

from ac_gan import BATCH_SIZE, DEVICE, EPOCHS, LEARNING_RATE, N_CLASSES, NOISE_DIM, TIME_SERIES_LENGTH, draw, get_dataloader, SERIES_TO_ENCODE
from sensor import SUPPORTED_SAMPLE_TYPES


class LSTMGenerator(nn.Module):
    """An LSTM based generator. It expects a sequence of noise vectors as input.
    Args:
        in_dim: Input noise dimensionality
        out_dim: Output dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms
    Input: noise of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, out_dim)
    """

    def __init__(self, out_dim, n_layers=1, hidden_dim=256):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.label_embeddings = nn.ModuleList()
        self.label_linears = nn.ModuleList()
        self.lstms = nn.ModuleList()
        self.output_linears = nn.ModuleList()

        for _ in range(len(SERIES_TO_ENCODE)):  # 生成器的输入是多个通道，分开处理以便适用所有情况
            temp_label_embedding = nn.Embedding(
                N_CLASSES, NOISE_DIM)
            self.label_embeddings.append(temp_label_embedding)

            FEATURE_DIM = 128

            temp_label_linear = nn.Sequential(
                nn.Linear(NOISE_DIM, FEATURE_DIM),
            )
            self.label_linears.append(temp_label_linear)

            temp_lstm = nn.LSTM(FEATURE_DIM, hidden_dim,
                                n_layers, batch_first=True)
            self.lstms.append(temp_lstm)

            temp_output_linear = nn.Sequential(
                nn.Linear(hidden_dim, out_dim)
            )
            self.output_linears.append(temp_output_linear)

    def forward(self, noise, labels):
        batch_size = labels.size(0)

        result = torch.tensor([], dtype=torch.float32)
        for index, lstm in enumerate(self.lstms):
            label_emb = self.label_embeddings[index](labels)
            temp_noise = noise[:, index, :]  # 按第二个维度取

            gen_input = torch.mul(label_emb, temp_noise)  # 和噪声融合

            out = self.label_linears[index](gen_input)  # 再经过一个线性层映射开
            out = out.unsqueeze(1)  # 增加一个维度

            h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
            c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

            recurrent_features, _ = lstm(out, (h_0, c_0))
            outputs = recurrent_features.contiguous().view(batch_size, self.hidden_dim)

            outputs = self.output_linears[index](outputs)
            result = torch.cat((result, outputs.unsqueeze(1)), dim=1)

        return result

    def generate(self, sample_type):
        noise = torch.randn(1, len(SERIES_TO_ENCODE), NOISE_DIM)
        index = SUPPORTED_SAMPLE_TYPES.index(sample_type)
        label = torch.tensor(np.array([index]), dtype=torch.long)
        result = self.forward(noise, label)
        result = result.detach().numpy()
        return result[0]


class LSTMDiscriminator(nn.Module):
    """An LSTM based discriminator. It expects a sequence as input and outputs a probability for each element. 
    Args:
        in_dim: Input timeseries dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms
    Inputs: sequence of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, 1)
    """

    def __init__(self, in_dim=1, n_layers=1, hidden_dim=256):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstms = nn.ModuleList()

        for _ in range(len(SERIES_TO_ENCODE)):
            temp = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim,
                           num_layers=n_layers, batch_first=True)
            self.lstms.append(temp)

        input_length = len(SERIES_TO_ENCODE)*hidden_dim
        self.classification = nn.Sequential(
            nn.Linear(input_length, input_length//2),
            nn.Linear(input_length//2, N_CLASSES),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        batch_size = input.size(0)

        result = torch.tensor([], dtype=torch.float32)
        for index, lstm in enumerate(self.lstms):
            h_0 = torch.zeros(self.n_layers, batch_size,
                              self.hidden_dim)  # lstm的hidden state权重矩阵
            c_0 = torch.zeros(self.n_layers, batch_size,
                              self.hidden_dim)  # lstm的cell state权重矩阵

            # 取第i个channel
            input_i = input[:, index, :]
            # [Batch Size, SeriesLength] -》 [Batch Size, SeriesLength, 1]
            input_i = input_i.unsqueeze(2)
            recurrent_features, _ = lstm(input_i, (h_0, c_0))
            # 取最后一个时刻的输出
            temp = recurrent_features[:, -1, :]
            # 将所有通道输出拼接起来 [Batch Size, lstm output dim*len(SERIES_TO_ENCODE)]
            result = torch.cat((result, temp), dim=1)
        outputs = self.classification(result)
        return outputs

    def classifiy(self, input):
        return self.forward(input)


generator = LSTMGenerator(TIME_SERIES_LENGTH)
discriminator = LSTMDiscriminator()


#mse_loss = nn.MSELoss().to(DEVICE)
ce_loss = nn.CrossEntropyLoss().to(DEVICE)

discriminator_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=LEARNING_RATE)
generator_optimizer = torch.optim.Adam(
    generator.parameters(), lr=LEARNING_RATE)


def train(dataloader):
    for epoch in range(EPOCHS):
        for i, (real_timeseries, label) in enumerate(dataloader, 0):
            batch_size = real_timeseries.size(0)
            ############################
            # (1) Update D network
            ###########################

            # Train with real data
            discriminator.zero_grad()
            real = real_timeseries.to(DEVICE)

            output = discriminator(real)
            errD_real = ce_loss(output, label)
            errD_real.backward(retain_graph=True)
            D_x = output.mean().item()

            # Train with fake data
            noise = torch.randn(batch_size, len(
                SERIES_TO_ENCODE), NOISE_DIM, device=DEVICE)
            fake_labels = torch.randint(
                0, N_CLASSES, (real_timeseries.size(0),), dtype=torch.long)
            one_hot_labels = torch.tensor(np.array([np.eye(N_CLASSES)[i]
                                                    for i in fake_labels]), dtype=torch.float32)

            fake_timeseries = generator(noise, fake_labels)
            output = discriminator(fake_timeseries)
            errD_fake = ce_loss(output, one_hot_labels)
            errD_fake.backward(retain_graph=True)

            D_G_z1 = output.mean().item()
            discriminator_optimizer.step()
            #errD = errD_real + errD_fake

            ############################
            # (2) Update G network
            ###########################
            if epoch % 3 == 0:
                generator.zero_grad()
                output = discriminator(fake_timeseries)
                errG = ce_loss(output, one_hot_labels)
                errG.backward()
                D_G_z2 = output.mean().item()

                generator_optimizer.step()

            # Report metrics
            print('[%d/%d][%d/%d] Loss_D_REAL: %.4f Loss_D_FAKE: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, EPOCHS, i, len(dataloader),
                     errD_real.item(), errD_fake.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        ##### End of the epoch #####

        """
        # Checkpoint
        if (epoch % opt.checkpoint_every == 0) or (epoch == (opt.epochs - 1)):
            torch.save(generator, '%s/%s_generator_epoch_%d.pth' %
                       (opt.outf, opt.run_tag, epoch))
            torch.save(discriminator, '%s/%s_netD_epoch_%d.pth' %
                       (opt.outf, opt.run_tag, epoch))
        """


if __name__ == "__main__":

    """noise = torch.randn(BATCH_SIZE, len(SERIES_TO_ENCODE), NOISE_DIM)
    print("Noise: ", noise.size())
    gen_out = generator(noise, torch.randint(
        0, N_CLASSES, (BATCH_SIZE,), dtype=torch.long))
    print("Generator output: ", gen_out.size())
    # gen_out = torch.randn(BATCH_SIZE, len(
    #    SERIES_TO_ENCODE), TIME_SERIES_LENGTH)
    dis_out = discriminator(gen_out)
    print("Discriminator output: ", dis_out.size())"""

    dataloader = get_dataloader(use_onehot=True)
    train(dataloader)

    type = "normal"
    t = generator.generate(type)
    print(t, t.shape)
    draw(t.flatten(), title=type)
