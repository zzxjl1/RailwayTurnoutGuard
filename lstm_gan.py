"""
基于lstm的gan，更适用于序列数据
"""

import torch
import torch.nn as nn

from ac_gan import EPOCHS, NOISE_DIM, TIME_SERIES_LENGTH


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

    def __init__(self, in_dim, out_dim, n_layers=1, hidden_dim=256):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh()
        )

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(
            batch_size*seq_len, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_len, self.out_dim)
        return outputs


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
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(
            batch_size*seq_len, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_len, 1)
        return outputs


generator = LSTMGenerator(NOISE_DIM, TIME_SERIES_LENGTH)
discriminator = LSTMDiscriminator(TIME_SERIES_LENGTH)


def train(dataloader):
    for epoch in range(EPOCHS):
        for i, data in enumerate(dataloader, 0):
            niter = epoch * len(dataloader) + i

            # Save just first batch of real data for displaying
            if i == 0:
                real_display = data.cpu()

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            # Train with real data
            netD.zero_grad()
            real = data.to(device)
            batch_size, seq_len = real.size(0), real.size(1)
            label = torch.full((batch_size, seq_len, 1),
                               real_label, device=device)

            output = netD(real)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with fake data
            noise = torch.randn(batch_size, seq_len, nz, device=device)
            if opt.delta_condition:
                # Sample a delta for each batch and concatenate to the noise for each timestep
                deltas = dataset.sample_deltas(
                    batch_size).unsqueeze(2).repeat(1, seq_len, 1)
                noise = torch.cat((noise, deltas), dim=2)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Visualize discriminator gradients
            for name, param in netD.named_parameters():
                writer.add_histogram(
                    "DiscriminatorGradients/{}".format(name), param.grad, niter)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()

            if opt.delta_condition:
                # If option is passed, alternate between the losses instead of using their sum
                if opt.alternate:
                    optimizerG.step()
                    netG.zero_grad()
                noise = torch.randn(batch_size, seq_len, nz, device=device)
                deltas = dataset.sample_deltas(
                    batch_size).unsqueeze(2).repeat(1, seq_len, 1)
                noise = torch.cat((noise, deltas), dim=2)
                # Generate sequence given noise w/ deltas and deltas
                out_seqs = netG(noise)
                delta_loss = opt.delta_lambda * delta_criterion(
                    out_seqs[:, -1] - out_seqs[:, 0], deltas[:, 0])
                delta_loss.backward()

            optimizerG.step()

            # Visualize generator gradients
            for name, param in netG.named_parameters():
                writer.add_histogram(
                    "GeneratorGradients/{}".format(name), param.grad, niter)

            ###########################
            # (3) Supervised update of G network: minimize mse of input deltas and actual deltas of generated sequences
            ###########################

            # Report metrics
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), end='')
            if opt.delta_condition:
                writer.add_scalar(
                    'MSE of deltas of generated sequences', delta_loss.item(), niter)
                print(' DeltaMSE: %.4f' %
                      (delta_loss.item()/opt.delta_lambda), end='')
            print()
            writer.add_scalar('DiscriminatorLoss', errD.item(), niter)
            writer.add_scalar('GeneratorLoss', errG.item(), niter)
            writer.add_scalar('D of X', D_x, niter)
            writer.add_scalar('D of G of z', D_G_z1, niter)

        ##### End of the epoch #####
        real_plot = time_series_to_plot(dataset.denormalize(real_display))
        if (epoch % opt.tensorboard_image_every == 0) or (epoch == (opt.epochs - 1)):
            writer.add_image("Real", real_plot, epoch)

        fake = netG(fixed_noise)
        fake_plot = time_series_to_plot(dataset.denormalize(fake))
        torchvision.utils.save_image(fake_plot, os.path.join(
            opt.imf, opt.run_tag+'_epoch'+str(epoch)+'.jpg'))
        if (epoch % opt.tensorboard_image_every == 0) or (epoch == (opt.epochs - 1)):
            writer.add_image("Fake", fake_plot, epoch)

        # Checkpoint
        if (epoch % opt.checkpoint_every == 0) or (epoch == (opt.epochs - 1)):
            torch.save(netG, '%s/%s_netG_epoch_%d.pth' %
                       (opt.outf, opt.run_tag, epoch))
            torch.save(netD, '%s/%s_netD_epoch_%d.pth' %
                       (opt.outf, opt.run_tag, epoch))


if __name__ == "__main__":

    noise = torch.randn(8, 16, NOISE_DIM)
    gen_out = generator(noise)
    dis_out = discriminator(gen_out)

    print("Noise: ", noise.size())
    print("Generator output: ", gen_out.size())
    print("Discriminator output: ", dis_out.size())
