import torch
from torch import nn


class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x):
        return x

class decoder(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x):
        return x

class generator(nn.Module):
    def __init__(self, embedding_dim, gen_dims, data_dim):
        super(generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(gen_dims):
            seq += [
                Residual(dim, item)
            ]
            dim += item
        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data

class discriminator(nn.Module):
    def __init__(self, input_dim, dis_dims, pack=10):
        super(discriminator, self).__init__()
        dim = input_dim * pack
        self.pack = pack
        self.packdim = dim
        seq = []
        for item in list(dis_dims):
            seq += [
                nn.Linear(dim, item),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5)
            ]
            dim = item
        seq += [nn.Linear(dim, 1)]
        self.seq = nn.Sequential(*seq)

    def forward(self, input):
        assert input.size()[0] % self.pack == 0
        return self.seq(input.view(-1, self.packdim))

class classifier(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x):
        return x