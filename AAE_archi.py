import torch
from scipy.stats import exponpow, cauchy, gamma, norm, rayleigh, expon, chi
from torch import exp
from torch.nn import ModuleDict, Linear, LeakyReLU, BatchNorm1d, Module, Sigmoid, Sequential, Tanh, Dropout, Softmax, MSELoss, CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from data import main_u

in_out = 26
z_dim = 15

cuda = True if torch.cuda.is_available() else False
label_dim = 1




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
        self.h1 = 25
        self.h2 = 22
        self.h3 = 15

        seq = [

            Linear(dim, self.h1),
            LeakyReLU(),
            BatchNorm1d(self.h1),
            Linear(self.h1, self.h1),
            LeakyReLU(),
            BatchNorm1d(self.h1),
            Linear(self.h1, self.h2),
            LeakyReLU(),
            BatchNorm1d(self.h2),
            Linear(self.h2, self.h2),
            LeakyReLU(),
            BatchNorm1d(self.h2),
            Linear(self.h2, self.h3),
            LeakyReLU(),
            BatchNorm1d(self.h3),
            Linear(self.h3, dim),
            LeakyReLU(),
            BatchNorm1d(dim)
               ]
        self.seq = Sequential(*seq)
        self.mu = Linear(dim, z_dim)
        self.logvar = Linear(dim, z_dim)


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
        self.h1 = 15
        self.h2 = 22
        self.h3 = 25

        self.shared = Sequential(
            Linear(dim, self.h1),
            LeakyReLU(),
            BatchNorm1d(self.h1),
            Linear(self.h1, self.h2),
            LeakyReLU(),
            BatchNorm1d(self.h2),
            Linear(self.h2, self.h3),
            LeakyReLU(),
            BatchNorm1d(self.h3),
            Linear(self.h3, self.h3),
            LeakyReLU(),
            BatchNorm1d(self.h3),
            Linear(self.h3, in_out),
            LeakyReLU(),
            BatchNorm1d(in_out)
        )

        self.discrete_out = {feature: Linear(in_out, num_classes)
                             for feature, num_classes in discrete_features.items()}
        self.continuous_out = {feature: Linear(in_out, 1)
                               for feature in continuous_features}

        self.discrete_out = ModuleDict(self.discrete_out)
        self.continuous_out = ModuleDict(self.continuous_out)

        self.ce = CrossEntropyLoss()
        self.mse = MSELoss()
        self.softmax = Softmax(dim=-1)
        self.tanh = Tanh()

    def forward(self, output):
        x = self.shared(output)
        return x

    def disc_cont(self, x):
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
        discrete_targets, continuous_targets = targets
        total_loss = 0
        for feature in self.discrete_features:
            if feature in targets:
                total_loss += self.ce(discrete_outputs[feature], discrete_targets[feature])
        for feature in self.continuous_features:
            if feature in targets:
                total_loss += self.mse(continuous_outputs[feature], continuous_targets[feature])

        return total_loss





class Discriminator(Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.h1 = 10
        self.h2 = 10
        self.h3 = 5
        # self.h4 = 5
        seq = [
            Linear(dim, self.h1),
            LeakyReLU(),
            Linear(self.h1, self.h2),
            LeakyReLU(),
            Dropout(0.2),
            Linear(self.h2, self.h3),
            LeakyReLU(),
            Attention(self.h3, 3),
            # Linear(self.h3, self.h4),
            # LeakyReLU(),
            # Attention(self.h4, 3),
            Linear(self.h3, 1),
            Sigmoid()]
        self.seq = Sequential(*seq)
    def forward(self, x):
        x = self.seq(x)
        return x


discrete = {"proto": 132,
            "service": 13,
            "state": 7,
            "is_ftp_login": 4,
            "ct_flw_http_mthd": 11,
            'ct_state_ttl': 16,
}

continuous = [feature for feature in main_u.X_train.columns if feature not in discrete]

encoder_generator = EncoderGenerator(in_out, ).cuda() if cuda else EncoderGenerator(in_out, )
decoder = Decoder(z_dim+ label_dim, discrete, continuous).cuda() if cuda else Decoder(z_dim+ label_dim, discrete, continuous)
discriminator = Discriminator(z_dim, ).cuda() if cuda else Discriminator(z_dim, )


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


X_train, X_val, y_train, y_val = main_u.vertical_split(main_u.X_train_sc, main_u.y_train)
dataset = CustomDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
val = CustomDataset(X_val, y_val)
val_dl = DataLoader(val, batch_size=6)
test = CustomDataset(main_u.X_test_sc, main_u.y_test)
test_dl = DataLoader(test, batch_size=32, shuffle=True, num_workers=4)


