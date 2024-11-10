import numpy as np
import torch
from scipy.stats import exponpow, cauchy, gamma, norm, rayleigh, expon, chi
from sklearn.model_selection import train_test_split
from torch import exp, normal
from torch.nn import Linear, LeakyReLU, BatchNorm1d, Module, Sigmoid, Sequential, Tanh, Dropout, Softmax
from torch.utils.data import DataLoader, Dataset, Subset
from data import main_u

in_out = 27
z_dim = 12

cuda = False
label_dim = 10

cont = ["id", "dur", "rate", "sload", "dload", "sinpkt", "dinpkt", "sjit", "djit", "stcpb", "dtcpd", "dwin", "tcprtt", "synack", "ackdat"]


def custom_dist(size):
    for i, _ in enumerate(main_u.X.columns):
        if i in [0, 8, 16, 23, 26]:
            z = exponpow(0.6, 0, 0.6).rvs(size)
        elif i in [2, 7, 18, 19]:
            z = cauchy(0.1, 0).rvs(size)
        elif i in [22]:
            z= gamma(0.4, 0, 0.2).rvs(size)
        elif i in [4]:
            z = norm(0.5, 0.1).rvs(size)
        elif i in [5, 6, 15, 20, 21]:
            z = rayleigh(0, 0).rvs(size)
        elif i in [3, 9]:
            z = chi(0.4, 0, 0.18).rvs(size)
        else:
            z = expon(0, 0).rvs(size)

        z = torch.tensor(z, dtype=torch.float32).cuda() if cuda else torch.tensor(z, dtype=torch.float32)
    return z


def reparameterization(mu, logvar, z_dim):
    std = exp(logvar / 2)
    # sampled_z = normal(0, 1, (mu.size(0), z_dim)).cuda() if cuda else normal(0, 1, (mu.size(0), z_dim))
    sampled_z = custom_dist((mu.size(0), z_dim)).cuda() if cuda else custom_dist((mu.size(0), z_dim))
    z = sampled_z * std + mu
    return z


class EncoderGenerator(Module):
    def __init__(self, in_out):
        super(EncoderGenerator, self).__init__()
        dim = in_out
        seq = [Linear(dim, 15),
               LeakyReLU(),
               BatchNorm1d(15),
               Dropout(0.2)]
        self.seq = Sequential(*seq)
        self.mu = Linear(15, z_dim)
        self.logvar = Linear(15, z_dim)


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
        seq = [Linear(dim, 15),
               LeakyReLU(),
               BatchNorm1d(15),
            Linear(15, in_out)
               ]
        self.seq = Sequential(*seq)
        self.softmax = Softmax(dim=-1)
        self.tanh = Tanh()

    def forward(self, x):
        if cont:
            return self.softmax(self.seq(x))
        else:
            return self.Tanh(self.seq(x))


class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        dim = z_dim
        seq = [
            Linear(dim, 3),
            LeakyReLU(),
            BatchNorm1d(3),
            Dropout(0.5),
            Linear(3, 1),
            Dropout(0.5),
            Sigmoid()]
        self.seq = Sequential(*seq)
    def forward(self, x):
        x = self.seq(x)
        return x

encoder_generator = EncoderGenerator(in_out, ).cuda() if cuda else EncoderGenerator(in_out, )
decoder = Decoder().cuda() if cuda else Decoder()
discriminator = Discriminator().cuda() if cuda else Discriminator()

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

def bootstrap_sample(dataset, n_samples):
    indices = np.random.choice(len(dataset), n_samples, replace=True)
    return Subset(dataset, indices)


X_train, X_val, y_train, y_val = train_test_split(main_u.X_train_sc, main_u.y_train, test_size=0.1, random_state=48)
dataset = CustomDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
val = CustomDataset(X_val, y_val)
val_dl = DataLoader(val, batch_size=1)
test = CustomDataset(main_u.X_test_sc, main_u.y_test)
test_dl = DataLoader(test, batch_size=32, shuffle=True, num_workers=4)
