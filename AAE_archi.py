import numpy as np
import torch
from scipy.stats import exponpow, cauchy, gamma, norm, rayleigh, expon, chi
from sklearn.model_selection import train_test_split
from torch import exp
from torch.nn import Linear, LeakyReLU, BatchNorm1d, Module, Sigmoid, Sequential, Tanh, Dropout, Softmax, MSELoss, CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from data import main_u

in_out = 26
z_dim = 18

cuda = False
label_dim = 10




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


class Attention(Module):
    def __init__(self, in_features, attention_size):
        super(Attention, self).__init__()
        self.attention_weights = Linear(in_features, attention_size)
        self.attention_score = Linear(attention_size, 1, bias=False)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        attn_weights = torch.tanh(self.attention_weights(x))
        attn_score = self.attention_score(attn_weights)
        attn_score = self.softmax(attn_score)

        weighted_input = x * attn_score
        return weighted_input


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
        seq = [Linear(dim, 22),
               LeakyReLU(),
               BatchNorm1d(22),
               Linear(22, 20),
               LeakyReLU(),
               BatchNorm1d(20),
               Linear(20, 19),
               LeakyReLU(),
               BatchNorm1d(19)

               ]
        self.seq = Sequential(*seq)
        self.mu = Linear(19, z_dim)
        self.logvar = Linear(19, z_dim)


    def forward(self, x):
        x = self.seq(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar, z_dim)
        return z


class Decoder(Module):
    def __init__(self, dim, discrete_features, continuous_features):
        super(Decoder, self).__init__()
        self.discrete_features = discrete_features
        self.continuous_features = continuous_features

        self.shared = Sequential(
            Linear(dim, 19),
            LeakyReLU(),
            BatchNorm1d(19)
        )

        self.discrete_out = {feature: Linear(19, num_classes)
                             for feature, num_classes in discrete_features.items()}
        self.continuous_out = {feature: Linear(19, 1)
                               for feature in continuous_features}

        self.discrete_out = torch.nn.ModuleDict(self.discrete_out)
        self.continuous_out = torch.nn.ModuleDict(self.continuous_out)

        self.ce = CrossEntropyLoss()
        self.mse = MSELoss()
        self.softmax = Softmax(dim=-1)
        self.tanh = Tanh()

    def forward(self, x):
        shared_features = self.shared(x)

        discrete_outputs = {}
        continuous_outputs = {}

        for feature in self.discrete_features:
            logits = self.discrete_out[feature](shared_features)
            discrete_outputs[feature] = self.softmax(logits)

        for feature in self.continuous_features:
            continuous_outputs[feature] = self.tanh(self.continuous_out[feature](shared_features))

        return discrete_outputs, continuous_outputs

    def compute_loss(self, outputs, targets):
        discrete_outputs, continuous_outputs = outputs
        total_loss = 0
        for feature in self.discrete_features:
            if feature in targets:
                total_loss += self.ce(discrete_outputs[feature], targets[feature])
        for feature in self.continuous_features:
            if feature in targets:
                total_loss += self.mse(continuous_outputs[feature], targets[feature])

        return total_loss


class Discriminator(Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        seq = [
            Linear(dim, 2),
            LeakyReLU(),
            Attention(2, 2),
            Linear(2, 1),
            LeakyReLU(),
            Dropout(0.5),
            Sigmoid()]
        self.seq = Sequential(*seq)
    def forward(self, x):
        x = self.seq(x)
        return x

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


X_train, X_val, y_train, y_val = train_test_split(main_u.X_train_sc, main_u.y_train, test_size=0.1, random_state=48)
dataset = CustomDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
val = CustomDataset(X_val, y_val)
val_dl = DataLoader(val, batch_size=6)
test = CustomDataset(main_u.X_test_sc, main_u.y_test)
test_dl = DataLoader(test, batch_size=32, shuffle=True, num_workers=4)

discrete = {"proto": 132,
            "service": 13,
            "state": 7,
            "is_ftp_login": 4,
            "ct_flw_http_mthd": 11}

continuous = ['id', 'dur', 'spkts', 'dpkts', 'rate',
       'sttl', 'dttl', 'sload', 'dload', 'sinpkt', 'dinpkt', 'sjit', 'djit',
       'swin', 'tcprtt', 'smean', 'dmean', 'trans_depth', 'response_body_len',
       'ct_srv_src', 'ct_state_ttl']


encoder_generator = EncoderGenerator(in_out, ).cuda() if cuda else EncoderGenerator(in_out, )
decoder = Decoder(z_dim+ label_dim, discrete, continuous).cuda() if cuda else Decoder(z_dim+ label_dim, discrete, continuous)
discriminator = Discriminator(z_dim, ).cuda() if cuda else Discriminator(z_dim, )
