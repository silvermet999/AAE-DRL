"""-----------------------------------------------import libraries-----------------------------------------------"""
import main
import argparse
import os
import numpy as np
import math
import itertools
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
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
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent code")
parser.add_argument("--input_size", type=int, default=32, help="size of each input dimension")
parser.add_argument("--channels", type=int, default=1, help="number of input channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between input sampling")
opt = parser.parse_args()
print(opt)
cuda = True if torch.cuda.is_available() else False

torch_gpu = torch.empty((20000, 20000)).cuda()
torch.cuda.memory_allocated()



"""-----------------------------------initialize variables for inputs and outputs-----------------------------------"""
input_shape_rs = main.X_pca_rs.shape
input_shape_mas = main.X_pca_mas.shape
nn_dim = 512
latent_dim = 10
lr = 0.005



"""-----------------------------------------------------classes-----------------------------------------------------"""
def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    """ 
    stockasticity :
    samples sampled_z from a standard normal distribution (np.random.normal(0, 1, ...)) with the same shape as mu; 
    the mean of the distribution in the latent space.
    Instead of directly using the mean mu to represent the latent variable, the model samples from a distribution around mu.
    
    Backpropagation:
    sampled_z seperate from mu and logvar.
    """
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z


# module compatible with PyTorch
class EncoderGenerator(nn.Module):
    def __init__(self):
        super(EncoderGenerator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(input_shape_rs)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # projects output to the dim of latent space
        self.mu = nn.Linear(nn_dim, opt.latent_dim)
        self.logvar = nn.Linear(nn_dim, opt.latent_dim)


# forward propagation
    def forward(self, input):
        # view() flatten
        input_flat = input.view(input.shape[0], -1)
        x = self.model(input_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, nn_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(input_shape_rs))),
            nn.Tanh(),
        )

    def forward(self, z):
        input_flat = self.model(z)
        input = input_flat.view(input_flat.shape[0], *input_shape_rs)
        return input


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, nn_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


# Use binary cross-entropy loss
adversarial_loss = torch.nn.BCELoss().cuda()
pixelwise_loss = torch.nn.L1Loss().cuda()



encoder_generator = EncoderGenerator().cuda()
# decoder = Decoder().cuda()
# discriminator = Discriminator().cuda()



"""_________________________________in case of more than one gpu (not the case here)_________________________________"""
def init_process():
    dist.init_process_group(
        backend='gloo',
        init_method='tcp://127.0.0.1:9000',
        rank=0,
        world_size=1
    )

def main():
    init_process()
    par.DataParallel(encoder_generator)
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
