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

        self.label_embedding = nn.Embedding(
            N_CLASSES, NOISE_DIM*len(SERIES_TO_ENCODE))
        self.label_linear = nn.Linear(NOISE_DIM, 768)

        self.lstm = nn.LSTM(768, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        batch_size = labels.size(0)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        label_emb = self.label_embedding(labels)
        label_emb = label_emb.view(
            batch_size, len(SERIES_TO_ENCODE), NOISE_DIM)
        gen_input = torch.mul(label_emb, noise)
        out = self.label_linear(gen_input)

        recurrent_features, _ = self.lstm(out, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(
            batch_size*len(SERIES_TO_ENCODE), self.hidden_dim))
        outputs = outputs.view(batch_size, len(SERIES_TO_ENCODE), self.out_dim)
        return outputs

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

    def __init__(self, in_dim, n_layers=1, hidden_dim=256):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim*len(SERIES_TO_ENCODE), N_CLASSES),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        batch_size = input.size(0)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        # 后两个维度合起来
        temp = recurrent_features.contiguous().view(
            batch_size, -1)
        outputs = self.linear(temp)
        return outputs

    def classifiy(self, input):
        return self.forward(input)


generator = LSTMGenerator(TIME_SERIES_LENGTH)
discriminator = LSTMDiscriminator(TIME_SERIES_LENGTH)

discriminator_loss = nn.BCELoss().to(DEVICE)
generator_loss = nn.MSELoss().to(DEVICE)

discriminator_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=LEARNING_RATE)
generator_optimizer = torch.optim.Adam(
    generator.parameters(), lr=LEARNING_RATE)

real_label = 1
fake_label = 0


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
            errD_real = discriminator_loss(output, label)
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
            errD_fake = discriminator_loss(output, one_hot_labels)

            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            errD.backward(retain_graph=True)
            discriminator_optimizer.step()

            ############################
            # (2) Update G network
            ###########################
            generator.zero_grad()
            output = discriminator(fake_timeseries)
            errG = discriminator_loss(output, one_hot_labels)
            errG.backward()
            D_G_z2 = output.mean().item()

            generator_optimizer.step()

            # Report metrics
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, EPOCHS, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

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
    dis_out = discriminator(gen_out)
    print("Discriminator output: ", dis_out.size())"""

    dataloader = get_dataloader(use_onehot=True)
    train(dataloader)

    type = "normal"
    t = generator.generate(type)
    print(t, t.shape)
    draw(t.flatten(), title=type)
