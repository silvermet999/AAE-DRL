import torch
from torch import exp, normal
from torch.nn import Linear, LeakyReLU, BatchNorm1d, Module, Sigmoid, Sequential, Tanh

in_out = 105
z_dim = 10
hl_dim = (100, 100,100,100)
hl_dimd = (10, 10,10,10,10,10,10,10,10,10)


def reparameterization(mu, logvar, z_dim):
    std = exp(logvar / 2)
    sampled_z = normal(0, 1, (mu.size(0), z_dim)).to("cuda:0")
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

class EncoderGenerator(Module):
    def __init__(self):
        super(EncoderGenerator, self).__init__()
        dim = in_out
        seq = []
        for i in list(hl_dim):
            seq += [hl_loop(dim, i)]
            dim += i
        seq += (Linear(dim, 100), LeakyReLU(0.2))
        self.seq = Sequential(*seq)
        # projects output to the dim of latent space
        self.mu = Linear(100, z_dim)
        self.logvar = Linear(100, z_dim)

    def forward(self, x):
        x = self.seq(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar, z_dim)
        return mu, logvar, z

class Decoder(Module):
    def __init__(self):
        super(Decoder, self).__init__()
        dim = z_dim
        seq = []
        for i in list(hl_dim):
            seq += [hl_loop(dim, i)]
            dim += i
        seq += (Linear(dim, in_out))
        self.seq = Sequential(*seq)
    def forward(self, x):
        return self.seq(x)


class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        dim = z_dim
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
    def forward(self, x):
        return self.seq(x)

encoder_generator = EncoderGenerator().to("cuda:0")
decoder = Decoder().to("cuda:1")
discriminator = Discriminator().to("cuda:1")