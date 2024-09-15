import torch
from torch import exp, normal
from torch.nn import Linear, LeakyReLU, BatchNorm1d, Module, Sigmoid, Sequential, Tanh, Dropout
from data import main

in_out = 102
z_dim = 128
hl_dim = (100, 150, 100)
hl_dime = (100, 150, 150, 100)
hl_dimd = (32, 32, 32, 12, 12)
cuda = True if torch.cuda.is_available() else False
encoded_tensor = torch.tensor(main.y_train, dtype=torch.float32).cuda() if cuda else torch.tensor(main.y_train, dtype=torch.float32)
label_dim = encoded_tensor.shape[1]


def reparameterization(mu, logvar, z_dim):
    std = exp(logvar / 2)
    sampled_z = normal(0, 1, (mu.size(0), z_dim)).cuda() if cuda else normal(0, 1, (mu.size(0), z_dim))
    z = sampled_z * std + mu
    return z

class hl_loop(Module):
    def __init__(self, i, o, _is_discriminator=False):
        super(hl_loop, self).__init__()
        self.fc1 = Linear(i, o)
        self.leakyrelu1 = LeakyReLU(0.2)
        self.bn = BatchNorm1d(o)
    def forward(self, l0):
        l1 = self.fc1(l0)
        l2 = self.leakyrelu1(l1)
        l3 = self.bn(l2)
        return torch.cat([l3, l0], dim=1)





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
                hl_loop(dim, i)]
            dim += i
        seq += [
            Dropout(0.1),
            Linear(dim, 1), Sigmoid()]
        self.seq = Sequential(*seq)
    def forward(self, x):
        return self.seq(x)

encoder_generator = EncoderGenerator(in_out, ).cuda() if cuda else EncoderGenerator(in_out, )
decoder = Decoder().cuda() if cuda else Decoder()
discriminator = Discriminator().cuda() if cuda else Discriminator()

# Average Reconstruction Loss: 0.3489 ± 0.0019
# Average Adversarial Loss: 3.4759 ± 0.0219