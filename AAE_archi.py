import math

import torch
from scipy.stats import lognorm, cauchy, norm, expon, gamma, uniform, chi2, exponpow
from torch import exp, normal
from torch.nn import Linear, LeakyReLU, BatchNorm1d, Module, Sigmoid, Sequential, Tanh, Dropout, Parameter, Softmax
from data import main

in_out = 111
z_dim = 128
hl_dim = (100, 150, 150, 150, 150)
hl_dime = (150, 200, 150, 100)
hl_dimd = (32, 32, 12, 6)
cuda = True if torch.cuda.is_available() else False
encoded_tensor = torch.tensor(main.y_train.values, dtype=torch.float32).cuda() if cuda else torch.tensor(main.y_train.values, dtype=torch.float32)
label_dim = encoded_tensor.shape[1]

def custom_dist(size):
    for i, _ in enumerate(main.X.columns):
        if i in [0, 6, 7, 10, 11, 101, 102, 103, 104, 106]:
            z = lognorm(0.5, -850, 2500).rvs(size)
        elif i in [1, 2, 53, 54, 55, 86, 97, 98, 99, 109]:
            z = cauchy(100, 10).rvs(size)
        elif i in [3, 9, 107]:
            z = norm(5447, 153).rvs(size)
        elif i in [4, 5, 13, 25, 34, 47, 49, 50, 51, 52, 60, 66, 69, 70]:
            z = expon(0, 8).rvs(size)
        elif i in [8, 95, 105]:
            z = chi2(1, 0, 13).rvs(size)
        elif i in [32, 58, 71]:
            z = gamma(1, 0, 20).rvs(size)
        elif i in [108, 111]:
            z = uniform(1, 29640).rvs(size)
        else:
            z = exponpow(0.1, 0, 0.95).rvs(size)

        z = torch.tensor(z, dtype=torch.float32).cuda() if cuda else torch.tensor(z, dtype=torch.float32)

    return z


class Self_Attn(Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, activation):
        """
        initialize self-attention layer
        initializes key, query and value layers, gamma coefficient and softmax layer

        Parameters
        ----------
        in_dim : int
            channel number of input
        activation : str
            activation type (e.g. 'relu')
        """
        super(Self_Attn, self).__init__()
        self.activation = activation
        self.query_conv = Linear(in_dim, (in_dim // 8)) # 50, 32, 256 => 50, 32, 32
        self.key_conv = Linear(in_dim, (in_dim // 8)) # 50, 32, 32
        self.value_conv = Linear(in_dim, in_dim) # # 50, 32, 256
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
        calculate key, query and value for each couple in input (number of couples N x N)

        Parameters
        ----------
        x : torch.Tensor
            input feature maps (shape: B X C X W X H)

        Returns
        -------
        out : torch.Tensor
            self attention value + input feature (shape: B X C X W X H)
        attention: torch.Tensor
            attention coefficients (shape: B X N X N) (N is W * H)
        """
        # batch, C, width, height = x.size()
        # proj_query = self.query_conv(x).view(batch, -1, width * height).permute(0, 2, 1)  # shape: B x N x (C // 8)
        print("x", x.shape)
        proj_query = self.query_conv(x)
        print("query", proj_query.shape)
        # proj_key = self.key_conv(x).view(batch, -1, width * height)  # shape: B x (C // 8) x N
        proj_key = self.key_conv(x)
        print("key", proj_key.shape)
        energy = torch.bmm(proj_query, proj_key)  # shape: B x N x N
        print("energy", energy.shape)
        attention = self.softmax(energy)  # shape: B x N x N
        print("att", attention.shape)
        # proj_value = self.value_conv(x).view(batch, -1, width * height)  # shape: B X C X N
        proj_value = self.value_conv(x)  # shape: B X C X N
        print("value", proj_value.shape)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # shape: B x C x N
        # out = out.view(batch, C, width, height)  # shape: B x C x W x H

        # calculate attention + input
        out = self.gamma * out + x  # shape: B x C x W x H
        return out, attention





def reparameterization(mu, logvar, z_dim):
    std = exp(logvar / 2)
    sampled_z = normal(0, 1, (mu.size(0), z_dim)).cuda() if cuda else normal(0, 1, (mu.size(0), z_dim))
    # sampled_z = exppower(loc=torch.tensor(0.), scale=torch.tensor(1.), b=0.1, size=(mu.size(0), z_dim)).cuda() if cuda else exppower(loc=torch.tensor(0.), scale=torch.tensor(1.), b=0.1, size=(mu.size(0), z_dim))
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
            Linear(dim, 100),
            LeakyReLU(0.2))
        self.seq = Sequential(*seq)
        # projects output to the dim of latent space
        self.mu = Linear(100, z_dim)
        self.logvar = Linear(100, z_dim)

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
                hl_loop(dim, i, _is_discriminator=True)]
            dim += i
        seq += [
            # Dropout(0.2),
            Linear(dim, 1), Sigmoid()]
        self.seq = Sequential(*seq)
    def forward(self, x):
        return self.seq(x)

encoder_generator = EncoderGenerator(in_out, ).cuda() if cuda else EncoderGenerator(in_out, )
decoder = Decoder().cuda() if cuda else Decoder()
discriminator = Discriminator().cuda() if cuda else Discriminator()

# Average Reconstruction Loss: 0.3489 ± 0.0019
# Average Adversarial Loss: 3.4759 ± 0.0219
