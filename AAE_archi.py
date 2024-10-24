import math

import torch
from scipy.stats import lognorm, cauchy, norm, expon, gamma, uniform, chi2, exponpow, rayleigh
from torch import exp, normal
from torch.nn import Linear, LeakyReLU, BatchNorm1d, Module, Sigmoid, Sequential, Tanh, Dropout, Softmax, ReLU
from data import main

in_out = 111
z_dim = 99
attention_dim=50
hl_dim = (100, 100)
hl_dime = (100, 100)
hl_dimd = (32, 32)
cuda = True if torch.cuda.is_available() else False
label_dim = 12

def custom_dist(size):
    for i, _ in enumerate(main.X.columns):
        if i in [0, 5, 7, 8, 10, 11, 95, 97, 101, 102, 103, 104, 105, 106]:
            z = lognorm(0.6, -0.9, 0.8).rvs(size)
        elif i in [1, 2, 4, 6, 49, 86, 99, 109]:
            z = cauchy(-0.3, 0.4).rvs(size)
        elif i in [3, 107]:
            z = norm(-0.15, 0.65).rvs(size)
        elif i in [5, 13, 25, 32, 34, 47, 50, 53, 54, 55, 60, 66, 69, 70, 71]:
            z = expon(0, 1).rvs(size)
        elif i in [51, 52, 100]:
            z = gamma(0.3, -2, 10).rvs(size)
        elif i in [109, 111]:
            z = uniform(1, 29640).rvs(size)
        elif i in [9]:
            z = rayleigh(-1.5, 1.1).rvs(size)
        else:
            z = exponpow(0.12, -8.41, 0.95).rvs(size)

        z = torch.tensor(z, dtype=torch.float32).cuda() if cuda else torch.tensor(z, dtype=torch.float32)

    return z


class AttentionLayer(Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.attention = Sequential(
            Linear(input_dim, attention_dim),
            ReLU(),
            Linear(attention_dim, input_dim),
            Softmax(dim=1)
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        weighted_output = x * attention_weights
        return weighted_output





def reparameterization(mu, logvar, z_dim):
    std = exp(logvar / 2)
    sampled_z = normal(0, 1, (mu.size(0), z_dim)).cuda() if cuda else normal(0, 1, (mu.size(0), z_dim))
    # sampled_z = custom_dist((mu.size(0), z_dim)).cuda() if cuda else custom_dist((mu.size(0), z_dim))
    z = sampled_z * std + mu
    return z

class hl_loop(Module):
    def __init__(self, i, o, _is_discriminator=False):
        super(hl_loop, self).__init__()
        self.fc1 = Linear(i, o)
        self.leakyrelu1 = LeakyReLU(0.2)
        self.bn = BatchNorm1d(o)
        self._is_discriminator = _is_discriminator
        self.dr = Dropout(0.1) if self._is_discriminator else None
    def forward(self, l0):
        l1 = self.fc1(l0)
        l2 = self.leakyrelu1(l1)
        l3 = self.bn(l2)
        l4 = self.dr(l3) if self.dr is not None else l3
        return torch.cat([l4, l0], dim=1)





class EncoderGenerator(Module):
    def __init__(self, in_out):
        super(EncoderGenerator, self).__init__()
        dim = in_out
        seq = []
        for i in list(hl_dime):
            seq += [hl_loop(dim, i)]
            dim += i
        seq += (
            Linear(dim, 46),
            LeakyReLU(0.2))
        self.seq = Sequential(*seq)
        self.mu = Linear(46, z_dim)
        self.logvar = Linear(46, z_dim)

    def forward(self, x):
        x = self.seq(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar, z_dim)
        return z

class Decoder(Module):
    def __init__(self):
        super(Decoder, self).__init__()
        dim = z_dim+ label_dim
        seq = []
        for i in list(hl_dim):
            seq += [hl_loop(dim, i)]
            dim += i
        seq += (Linear(dim, in_out), Tanh())
        self.seq = Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        dim = z_dim
        seq = []
        for i in list(hl_dim):
            seq += [
                hl_loop(dim, i)]
            dim += i
        # self.attention_layer = AttentionLayer(dim, attention_dim)

        seq += [
            Linear(dim, 1), Sigmoid()]
        self.seq = Sequential(*seq)
    def forward(self, x):
        # x = self.seq[:-2](x)
        # x = self.attention_layer(x)
        # x = self.seq[-2:](x)
        x = self.seq(x)
        return x

encoder_generator = EncoderGenerator(in_out, ).cuda() if cuda else EncoderGenerator(in_out, )
decoder = Decoder().cuda() if cuda else Decoder()
discriminator = Discriminator().cuda() if cuda else Discriminator()

# Average Reconstruction Loss: 0.3489 ± 0.0019
# Average Adversarial Loss: 3.4759 ± 0.0219
