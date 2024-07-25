"""-----------------------------------------------import libraries-----------------------------------------------"""
import os

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import MultiStepLR

import dim_reduction
import argparse
import numpy as np
import pandas as pd
import itertools
import torch
from torch.nn import BatchNorm1d, LeakyReLU, Linear, Module, Sequential, Tanh, Sigmoid, BCELoss, L1Loss
from torch import Tensor, cuda, exp
from torchsummary import summary
import mlflow
import main

from skopt import gp_minimize
from skopt.space import Real
import itertools
"""-----------------------------------initialize variables for inputs and outputs-----------------------------------"""
cuda = True if cuda.is_available() else False
df_train = main.x_train_rs[:10000]
df_test = main.x_test_rs[:2500]
in_out_rs = 127 # in for the enc/gen out for the dec
hl_dim = (100, 100, 100, 100, 100)
hl_dimd = (10, 10, 10, 10, 10, 10, 10, 10, 10, 10)
out_in_dim = 100 # in for the dec and disc out for the enc/gen
z_dim = 10
params = {
    "lr": 0.01,
    "batch_size": 24,
    "n_epochs": 100,
    "optimizer": "Adam",
    "gamma": 0.9
}



"""---------------------------------------------backprop and hidden layers-------------------------------------------"""
def reparameterization(mu, logvar, z_dim):
    std = exp(logvar / 2)
    device = mu.device
    log_normal = torch.distributions.LogNormal(loc=0, scale=1)

    # Sample from the distribution with the correct size
    sampled_z = log_normal.sample((mu.size(0), z_dim)).to(device)

    z = sampled_z * std + mu
    return z


class hl_loop(Module):
    def __init__(self, i, o):
        super(hl_loop, self).__init__()
        self.fc1 = Linear(i, o)
        self.leakyrelu1 = LeakyReLU(0.2)
        self.bn = BatchNorm1d(o)

    def forward(self, l0):
        l1 = self.fc1(l0)
        l2 = self.leakyrelu1(l1)
        l3= self.bn(l2)
        return torch.cat([l3, l0], dim=1)



"""----------------------------------------------------AAE blocks----------------------------------------------------"""
# module compatible with PyTorch
class EncoderGenerator(Module):
    def __init__(self):
        super(EncoderGenerator, self).__init__()
        dim = in_out_rs
        seq = []
        for i in list(hl_dim):
            seq += [hl_loop(dim, i)]
            dim += i
        seq.append(Linear(dim, out_in_dim))
        self.seq = Sequential(*seq)


        # projects output to the dim of latent space
        self.mu = Linear(out_in_dim, z_dim)
        self.logvar = Linear(out_in_dim, z_dim)


# forward propagation
    def forward(self, input_):
        x = self.seq(input_)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar, z_dim)
        return z


class Decoder(Module):
    def __init__(self):
        super(Decoder, self).__init__()
        dim = z_dim
        seq = []
        for i in list(hl_dim):
            seq += [hl_loop(dim, i)]
            dim += i
        seq += [Linear(dim, in_out_rs), Tanh()]
        self.seq = Sequential(*seq)

    def forward(self, z):
        input_ = self.seq(z)
        return input_



class Discriminator(Module):
    def __init__(self, pack=10):
        super(Discriminator, self).__init__()
        dim = z_dim * pack
        self.pack = pack
        self.packdim = dim
        seq = []
        for i in list(hl_dimd):
            seq += [
                Linear(z_dim, i),
                LeakyReLU(0.2, inplace=True),
                Linear(10, 10),
                LeakyReLU(0.2, inplace=True),
            ]
            dim = i
        seq += [Linear(dim, 1), Sigmoid()]
        self.seq = Sequential(*seq)


    def forward(self, input_):
        return self.seq(input_)



"""--------------------------------------------------loss and optim--------------------------------------------------"""
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("MLflow Quickstart")


def objective(params):
    lr_D, beta1_D, beta2_D, lr_G, beta1_G, beta2_G = params
    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                   lr=lr_D, betas=(beta1_D, beta2_D))
    optimizer_G = torch.optim.Adam(
        itertools.chain(encoder_generator.parameters(), decoder.parameters()),
        lr=lr_G, betas=(beta1_G, beta2_G))
    for epoch in range(10):
        batch_data = df_train[:10000]
        train_data_tensor = torch.tensor(batch_data, dtype=torch.float).cuda() if cuda else torch.tensor(batch_data,
                                                                                                         dtype=torch.float)

        real = (train_data_tensor - train_data_tensor.mean()) / train_data_tensor.std()
        valid = torch.ones((train_data_tensor.shape[0], 1)).cuda() if cuda else torch.ones(
            (train_data_tensor.shape[0], 1))
        fake = torch.zeros((train_data_tensor.shape[0], 1)).cuda() if cuda else torch.zeros(
            (train_data_tensor.shape[0], 1))

        optimizer_G.zero_grad()
        encoded = encoder_generator(real)
        decoded = decoder(encoded)
        g_loss = 0.01 * adversarial_loss(discriminator(encoded), valid) + 0.99 * recon_loss(decoded, real)

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        log_normal = torch.distributions.LogNormal(loc=0, scale=1)
        z = log_normal.sample((batch_data.shape[0], z_dim)).to("cuda")
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

    return d_loss.item(), g_loss.item()


adversarial_loss = BCELoss().cuda() if cuda else BCELoss()
recon_loss = L1Loss().cuda() if cuda else L1Loss()

encoder_generator = EncoderGenerator().cuda() if cuda else EncoderGenerator()
summary(encoder_generator, input_size=(in_out_rs,))
decoder = Decoder().cuda() if cuda else Decoder()
summary(decoder, input_size=(z_dim,))
discriminator = Discriminator().cuda() if cuda else Discriminator()
summary(discriminator, input_size=(z_dim,))

space = [
    Real(1e-5, 1e-2, name='lr_D', prior='log-uniform'),
    Real(0.5, 0.99, name='beta1_D'),
    Real(0.9, 0.999, name='beta2_D'),
    Real(1e-5, 1e-2, name='lr_G', prior='log-uniform'),
    Real(0.5, 0.99, name='beta1_G'),
    Real(0.9, 0.999, name='beta2_G')
]

res = gp_minimize(objective, space, n_calls=100, random_state=0)

best_lr_D, best_beta1_D, best_beta2_D, best_lr_G, best_beta1_G, best_beta2_G = res.x
