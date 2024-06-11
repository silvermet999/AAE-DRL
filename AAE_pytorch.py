"""-----------------------------------------------import libraries-----------------------------------------------"""
import main
import argparse
import numpy as np
import pandas as pd
import math
import itertools
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch import Tensor, cuda, exp
from torchsummary import summary
from torch.nn import parallel as par
from torch import distributed as dist



"""-----------------------------------------------command-line options-----------------------------------------------"""
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--z_dim", type=int, default=10, help="dimensionality of the latent code")
parser.add_argument("--input_size", type=int, default=32, help="size of each input dimension")
parser.add_argument("--channels", type=int, default=1, help="number of input channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between input sampling")
opt = parser.parse_args()
print(opt)
cuda = True if cuda.is_available() else False

# torch_gpu = torch.empty((20000, 20000)).cuda()
# torch.cuda.memory_allocated()



"""-----------------------------------initialize variables for inputs and outputs-----------------------------------"""
# input_shape_rs = main.X_pca_rs.shape
# input_shape_mas = main.X_pca_mas.shape
df_sel = main.df.iloc[:500, :100]
input_shape_rs = 100
nn_dim = 500
z_dim = 10
lr = 0.005



"""-----------------------------------------------------classes-----------------------------------------------------"""
def reparameterization(mu, logvar):
    std = exp(logvar / 2)
    """ 
    stockasticity :
    samples sampled_z from a standard normal distribution (np.random.normal(0, 1, ...)) with the same shape as mu; 
    the mean of the distribution in the latent space.
    Instead of directly using the mean mu to represent the latent variable, the model samples from a distribution around mu.
    
    Backpropagation:
    sampled_z seperate from mu and logvar.
    """
    sampled_z = Tensor(np.random.normal(0, 1, (mu.size(0), opt.z_dim)))
    z = sampled_z * std + mu
    return z


# module compatible with PyTorch
class EncoderGenerator(nn.Module):
    def __init__(self):
        super(EncoderGenerator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(input_shape_rs)), 500),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # projects output to the dim of latent space
        self.mu = nn.Linear(nn_dim, opt.z_dim)
        self.logvar = nn.Linear(nn_dim, opt.z_dim)


# forward propagation
    def forward(self, input_flat):
        # view() flatten
        # input_flat = input.view(input.shape[0], -1)
        x = self.model(input_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.z_dim, nn_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(500, int(np.prod(input_shape_rs))),
            nn.Tanh(),
        )

    def forward(self, z):
        input_flat = self.model(z)
        # input = input_flat.view(input_flat.shape[0], *input_shape_rs)
        return input_flat


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.z_dim, nn_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(500, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


# Use binary cross-entropy loss
adversarial_loss = nn.BCELoss().cuda() if cuda else nn.BCELoss()
pixelwise_loss = nn.L1Loss().cuda() if cuda else nn.L1Loss()



encoder_generator = EncoderGenerator().cuda() if cuda else EncoderGenerator()
decoder = Decoder().cuda() if cuda else Decoder()
discriminator = Discriminator().cuda() if cuda else Discriminator()



optimizer_G = torch.optim.Adam(
    itertools.chain(encoder_generator.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_runs(n_row, batches_done):
    # Sample noise
    z = Tensor(np.random.normal(0, 1, (n_row ** 2, opt.z_dim)))
    gen_input = decoder(z)
    gen_data = gen_input.data.cuda().numpy() if cuda else gen_input.data.numpy()
    df = pd.DataFrame(gen_data)
    df.to_csv(f"runs/{batches_done}.csv", index=False)


data_tensor = torch.tensor(df_sel.values, dtype=torch.float)
valid = torch.ones((data_tensor.shape[0], 1))
fake = torch.zeros((data_tensor.shape[0], 1))


for epoch in range(opt.n_epochs):
    # Configure input
    real = data_tensor
    optimizer_G.zero_grad()

    encoded = encoder_generator(real)
    decoded = decoder(encoded)
    g_loss = 0.001 * adversarial_loss(discriminator(encoded), valid) + 0.999 * pixelwise_loss(
                decoded, real
            )

    g_loss.backward()
    optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()

    # Sample noise as discriminator ground truth
    z = Tensor(np.random.normal(0, 1, (data_tensor.shape[0], opt.z_dim)))

    # Measure discriminator's ability to classify real from generated samples
    real_loss = adversarial_loss(discriminator(z), valid)
    fake_loss = adversarial_loss(discriminator(encoded.detach()), fake)
    d_loss = 0.5 * (real_loss + fake_loss)

    d_loss.backward()
    optimizer_D.step()

    print(
        epoch, opt.n_epochs, len(df_sel), d_loss.item(), g_loss.item()
    )

    batches_done = epoch * len(df_sel)
    if batches_done % opt.sample_interval == 0:
        sample_runs(n_row=5, batches_done=batches_done)





