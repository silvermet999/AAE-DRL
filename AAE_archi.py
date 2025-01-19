import torch
from scipy.stats import cauchy, gamma, rayleigh, expon, chi2, uniform, exponpow, lognorm, norm
from torch import exp
from torch.nn import ModuleDict, Linear, LeakyReLU, BatchNorm1d, Module, Sigmoid, Sequential, Tanh, Dropout, Softmax, \
    MSELoss, CrossEntropyLoss, BCELoss
from torch.utils.data import Dataset
from data import main_u

in_out = 30
z_dim = 15

cuda = True if torch.cuda.is_available() else False
label_dim = 4




def custom_dist(size):
    for i, _ in enumerate(main_u.X.columns):
        if i in [22]:
            z = lognorm(1, 0, 0).rvs(size)
        elif i in [0, 27, 28, 29]:
            z = chi2(0.8, 0, 0.4).rvs(size)
        elif i in [1, 6, 7, 8, 9, 11, 12, 16, 23]:
            z = rayleigh(0, 0).rvs(size)
        elif i in [17, 18, 24, 26]:
            z= gamma(0.1, 0, 0.2).rvs(size)
        elif i in [3]:
            z=norm(1.4, 0.8).rvs(size)
        elif i in [2, 25]:
            z = exponpow(1.5, 0.1, 1).rvs(size)
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
    sampled_z = custom_dist((mu.size(0), z_dim)).cuda() if cuda else custom_dist((mu.size(0), z_dim))
    z = sampled_z * std + mu
    return z


class EncoderGenerator(Module):
    def __init__(self, in_out):
        super(EncoderGenerator, self).__init__()
        dim = in_out
        self.h1 = 20
        # self.h2 = 20

        seq = [

            Linear(dim, self.h1),
            LeakyReLU(),
            BatchNorm1d(self.h1),
            Dropout(0.5),
            Linear(self.h1, self.h1),
            LeakyReLU(),
            BatchNorm1d(self.h1),
            # Linear(self.h1, self.h2),
            # LeakyReLU(),
            # BatchNorm1d(self.h2),
            Linear(self.h1, dim),
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
    def __init__(self, dim, discrete_features, continuous_features, binary_features):
        super(Decoder, self).__init__()
        self.dim = dim
        self.discrete_features = discrete_features
        self.continuous_features = continuous_features
        self.binary_features = binary_features
        self.h1 = 25
        self.h2 = 20

        self.shared = Sequential(
            Linear(self.dim, self.h1),
            LeakyReLU(),
            BatchNorm1d(self.h1),
            Dropout(0.3),
            Linear(self.h1, self.h2),
            LeakyReLU(),
            BatchNorm1d(self.h2),
            Dropout(0.2),
            Linear(self.h2, in_out),
            LeakyReLU(),
            BatchNorm1d(in_out)
        )

        self.discrete_out = {feature: Linear(in_out, num_classes)
                             for feature, num_classes in discrete_features.items()}
        self.continuous_out = {feature: Linear(in_out, 1)
                               for feature in continuous_features}
        self.binary_out = {feature: Linear(in_out, 2)
                                for feature in binary_features}

        self.discrete_out = ModuleDict(self.discrete_out)
        self.continuous_out = ModuleDict(self.continuous_out)
        self.binary_out = ModuleDict(self.binary_out)

        self.ce = CrossEntropyLoss()
        self.mse = MSELoss()
        self.bce = BCELoss()
        self.softmax = Softmax(dim=-1)
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()

    def disc_cont(self, x):
        shared_features = self.shared(x)

        discrete_outputs = {}
        continuous_outputs = {}
        binary_outputs = {}

        for feature in self.discrete_features:
            logits = self.discrete_out[feature](shared_features)
            discrete_outputs[feature] = self.softmax(logits)

        for feature in self.continuous_features:
            continuous_outputs[feature] = self.tanh(self.continuous_out[feature](shared_features))

        for feature in self.binary_features:
            binary_outputs[feature] = self.sigmoid(self.binary_out[feature](shared_features))

        return discrete_outputs, continuous_outputs, binary_outputs



    def compute_loss(self, outputs, targets):
        discrete_outputs, continuous_outputs, binary_outputs = outputs
        discrete_targets, continuous_targets, binary_targets = targets
        total_loss = 0
        for feature in self.discrete_features:
            if feature in targets:
                total_loss += self.ce(discrete_outputs[feature], discrete_targets[feature])
        for feature in self.continuous_features:
            if feature in targets:
                total_loss += self.mse(continuous_outputs[feature], continuous_targets[feature])
        for feature in self.binary_features:
            if feature in targets:
                total_loss += self.bce(binary_outputs[feature], binary_targets[feature])

        return total_loss





class Discriminator(Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.h1 = 12
        self.h2 = 10
        self.h3 = 8

        seq = [
            Linear(dim, self.h1),
            LeakyReLU(),
            Linear(self.h1, self.h1),
            LeakyReLU(),
            Linear(self.h1, self.h2),
            LeakyReLU(),
            Linear(self.h2, self.h2),
            LeakyReLU(),
            Linear(self.h2, self.h3),
            LeakyReLU(),
            Linear(self.h3, self.h3),
            LeakyReLU(),
            # Attention(self.h3, 3),
            Linear(self.h3, self.h3),
            LeakyReLU(),
            Linear(self.h3, 1),
            Sigmoid()]
        self.seq = Sequential(*seq)
    def forward(self, x):
        x = self.seq(x)
        return x


discrete = {
            "state": 8,
            # "service": 13,
            "ct_state_ttl": 7,
            # "dttl": 9,
            # "sttl": 13,
            "trans_depth": 11
}

binary = ["proto", "is_ftp_login"]
discrete_and_binary = set(discrete.keys()).union(set(binary))
continuous = [feature for feature in main_u.X_train.columns if feature not in discrete_and_binary]

encoder_generator = EncoderGenerator(in_out, ).cuda() if cuda else EncoderGenerator(in_out, )
decoder = Decoder(z_dim+label_dim, discrete, continuous, binary).cuda() if cuda else (
    Decoder(z_dim+label_dim, discrete, continuous, binary))
discriminator = Discriminator(z_dim, ).cuda() if cuda else Discriminator(z_dim, )


def types_append(discrete_out, continuous_out, binary_out, discrete_samples, continuous_samples, binary_samples):
    for feature in decoder.discrete_features:
        discrete_samples[feature].append(torch.argmax(torch.round(discrete_out[feature]), dim=-1))

    for feature in decoder.continuous_features:
        continuous_samples[feature].append(continuous_out[feature])

    for feature in decoder.binary_features:
        binary_samples[feature].append(torch.argmax(torch.round(binary_out[feature]), dim=-1))
    return discrete_samples, continuous_samples, binary_samples

def type_concat(discrete_samples, continuous_samples, binary_samples):
    for feature in decoder.discrete_features:
        discrete_samples[feature] = torch.cat(discrete_samples[feature], dim=0)

    for feature in decoder.continuous_features:
        continuous_samples[feature] = torch.cat(continuous_samples[feature], dim=0)

    for feature in decoder.binary_features:
        binary_samples[feature] = torch.cat(binary_samples[feature], dim=0)

    return discrete_samples, continuous_samples, binary_samples




class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label


def dataset_function(dataset, batch_size_t, batch_size_o, train=True):
    total_size = len(dataset)
    test_size = total_size // 5
    val_size = total_size // 10
    train_size = total_size - (test_size + val_size)
    train_subset = torch.utils.data.Subset(dataset, range(train_size))
    val_subset = torch.utils.data.Subset(dataset,
                                         range(train_size, train_size + val_size))
    test_subset = torch.utils.data.Subset(dataset,
                                          range(train_size + val_size, total_size))
    if train:
        train_loader = torch.utils.data.DataLoader(train_subset,
                                               batch_size=batch_size_t,
                                               shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset,
                                             batch_size=batch_size_o,
                                             shuffle=False)
        return train_loader, val_loader

    else:
        test_loader = torch.utils.data.DataLoader(test_subset,
                                              batch_size=batch_size_o,
                                              shuffle=False)

        return test_loader

dataset = CustomDataset(main_u.X_train_sc.to_numpy(), main_u.y_train.to_numpy())

