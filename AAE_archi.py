"""-----------------------------------------------import libraries-----------------------------------------------"""
import os

import torch
from torch.nn import BatchNorm1d, LeakyReLU, Linear, Module, Sequential, Tanh, Sigmoid, Dropout, Conv1d
from torch import cuda, exp

"""-----------------------------------initialize variables for inputs and outputs-----------------------------------"""
# if unsupervised in_out is 105 else 106
# if clean and unsupervised  else
cuda = True if cuda.is_available() else False
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
in_out = 105 # in for the enc/gen out for the dec
# out_in_dim = 100 # in for the dec and disc out for the enc/gen
z_dim = 32





"""---------------------------------------------backprop and hidden layers-------------------------------------------"""
def reparameterization(mu, logvar, z_dim):
    std = exp(logvar / 2)
    device = mu.device
    log_normal = torch.distributions.LogNormal(loc=0, scale=1)
    sampled_z = log_normal.sample((mu.size(0), z_dim)).to(device)
    z = sampled_z * std + mu
    return z



"""----------------------------------------------------AAE blocks----------------------------------------------------"""


class EncoderGenerator(Module):
    def __init__(self):
        super(EncoderGenerator, self).__init__()
        self.seq = Sequential(
            Linear(in_out, 64),
            LeakyReLU(0.1, inplace=True),
            BatchNorm1d(64),
            Linear(64, 128),
            LeakyReLU(0.1, inplace=True),
            BatchNorm1d(128),
            Linear(128, 256),
            LeakyReLU(0.1, inplace=True),
            BatchNorm1d(256),
            Linear(256, 128),
            LeakyReLU(0.2, inplace=True),
            BatchNorm1d(128),
            Linear(128, 64),
            LeakyReLU(0.3, inplace=True),
            BatchNorm1d(64),
            Linear(64, 105),
        )

        # Latent space projections
        self.mu = Linear(in_out, z_dim)
        self.logvar = Linear(in_out, z_dim)

    def forward(self, x):
        x = self.seq(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar, mu.size(1))
        return z


class Decoder(Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.seq = Sequential(
            Linear(z_dim, 32),
            LeakyReLU(0.1, inplace=True),
            BatchNorm1d(32),
            Linear(32, 64),
            LeakyReLU(0.2, inplace=True),
            BatchNorm1d(64),
            Linear(64, 128),
            LeakyReLU(0.2, inplace=True),
            BatchNorm1d(128),
            Linear(128, 256),
            LeakyReLU(0.3, inplace=True),
            BatchNorm1d(256),
            Linear(256, 128),
            LeakyReLU(0.3, inplace=True),
            BatchNorm1d(128),
            Linear(128, 105),
            Tanh()
        )

    def forward(self, z):
        input_ = self.seq(z)
        return input_




class Discriminator(Module):
    def __init__(self, pack=10):
        super(Discriminator, self).__init__()
        dim = z_dim * pack
        self.pack = pack
        self.packdim = dim
        self.seq = Sequential(
            Linear(32, 64),
            LeakyReLU(0.1, inplace=True),
            BatchNorm1d(64),
            Linear(64, 128),
            LeakyReLU(0.2, inplace=True),
            BatchNorm1d(128),
            Dropout(0.1),
            Linear(128, 64),
            LeakyReLU(0.2, inplace=True),
            BatchNorm1d(64),
            Linear(64, 32),
            LeakyReLU(0.2, inplace=True),
            BatchNorm1d(32),
            Linear(32, 1),
            Sigmoid()
        )


    def forward(self, input_):
        return self.seq(input_)