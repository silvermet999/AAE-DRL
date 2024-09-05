import torch
from torch import exp, normal
from torch.nn import Linear, LeakyReLU, BatchNorm1d, Module, Sigmoid, Sequential, Tanh
from AAE import main

in_out = 105
z_dim = 32
hl_dim = (100, 200, 300, 300, 200, 100)
hl_dimd = (32, 64, 128, 256, 512, 256, 128, 64, 32)
cuda = True if torch.cuda.is_available() else False
encoded_tensor = torch.tensor(main.y.values, dtype=torch.float32).cuda() if cuda else torch.tensor(main.y.values, dtype=torch.float32)
label_dim = encoded_tensor.shape[1]



def reparameterization(mu, logvar, z_dim):
    std = exp(logvar / 2)
    sampled_z = normal(0, 1, (mu.size(0), z_dim)).cuda() if cuda else normal(0, 1, (mu.size(0), z_dim))
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
            seq += [hl_loop(dim, i)]
            dim += i
        # for i in list(hl_dim):
            # seq += [
            #     Linear(z_dim, i),
            #     LeakyReLU(0.2, inplace=True),
            #     # Linear(32, i),
            #     # LeakyReLU(0.2, inplace=True),
            # ]
            # dim = i
        seq += [Linear(dim, 1), Sigmoid()]
        self.seq = Sequential(*seq)
    def forward(self, x):
        return self.seq(x)

encoder_generator = EncoderGenerator().cuda() if cuda else EncoderGenerator()
decoder = Decoder().cuda() if cuda else Decoder()
discriminator = Discriminator().cuda() if cuda else Discriminator()

# {'lr': 0.00016884384789265117, 'beta1': 0.8655422367267052, 'beta2': 0.9091518354520545, 'lrd': 1.753093187018987e-05, 'beta1d': 0.7845178698842424, 'beta2d': 0.9528948326902217}. 
