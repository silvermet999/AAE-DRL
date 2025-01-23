import pickle

import pandas as pd
from scipy.stats import cauchy, gamma, rayleigh, expon, chi2, uniform, exponpow, lognorm, norm
from torch.utils.data import Dataset

import numpy as np
import torch
from data import main_u

cuda = True if torch.cuda.is_available() else False



def df_type_split(df):
    X_cont = df.drop(["proto", "trans_depth", 'state', 'ct_state_ttl', "is_ftp_login",
                      # 'service', 'dttl', "is_sm_ips_ports", "ct_ftp_cmd", "ct_flw_http_mthd", 'sttl',
                      ], axis=1)
    X_disc = df[["proto", "trans_depth", 'state', 'ct_state_ttl', "is_ftp_login",
                      # 'service', 'dttl', "is_sm_ips_ports", "ct_ftp_cmd", "ct_flw_http_mthd", 'sttl',
                      ]]
    return X_disc, X_cont

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

discrete = {
            "state": 5,
            # "service": 13,
            "ct_state_ttl": 6,
            # "dttl": 9,
            # "sttl": 13,
            "trans_depth": 11
}

binary = ["proto", "is_ftp_login"]
discrete_and_binary = set(discrete.keys()).union(set(binary))
continuous = [feature for feature in main_u.X_train.columns if feature not in discrete_and_binary]


def types_append(decoder, discrete_out, continuous_out, binary_out, discrete_samples, continuous_samples, binary_samples):
    for feature in decoder.discrete_features:
        discrete_samples[feature].append(torch.argmax(torch.round(discrete_out[feature]), dim=-1))

    for feature in decoder.continuous_features:
        continuous_samples[feature].append(continuous_out[feature])

    for feature in decoder.binary_features:
        binary_samples[feature].append(torch.argmax(torch.round(binary_out[feature]), dim=-1))
    return discrete_samples, continuous_samples, binary_samples

def type_concat(decoder, discrete_samples, continuous_samples, binary_samples):
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


def inverse_sc_cont(X, synth):
    max_abs_values = np.abs(X).max(axis=0)
    synth_inv = synth * max_abs_values
    return pd.DataFrame(synth_inv, columns=X.columns, index=synth.index)



class RL_dataloader:
    def __init__(self, dataloader):
        self.loader = dataloader
        self.loader_iter = iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def next_data(self):
        try:
            data, label = next(self.loader_iter)

        except:
            self.loader_iter = iter(self.loader)
            data, label = next(self.loader_iter)

        return data, label




class ReplayBuffer(object):
    def __init__(self):
        self.storage = []
        self._saved = []
        self._sample_ind = None
        self._ind_to_save = 0

    def add(self, data):
        self.storage.append(data)
        self._saved.append(False)

    def sample(self):
        ind = np.random.randint(0, len(self.storage))
        self._sample_ind = ind
        return self[ind]

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, items):
        if hasattr(items, '__iter__'):
            items_iter = items
        else:
            items_iter = [items]

        s, a1, a2, n, r, d, t = [], [], [], [], [], [], []
        for i in items_iter:
            S, A1, A2, N, R, D, T = self.storage[i]
            s.append(np.array(S, copy=False))
            a1.append(np.array(A1, copy=False))
            a2.append(np.array(list(A2.items()), copy=False))
            n.append(np.array(N, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
            t.append(np.array(T, copy=False))

        return (np.array(s).squeeze(0), np.array(a1).squeeze(0), np.array(a2).squeeze(0),
                np.array(n).squeeze(0), np.array(r).squeeze(0), np.array(d).squeeze(0).reshape(-1, 1), np.array(t).squeeze(0))



def all_samples(discrete_samples, continuous_samples, binary_samples):
    discrete_tensors = list(discrete_samples.values())
    continuous_tensors = list(continuous_samples.values())
    binary_tensors = list(binary_samples.values())

    all_tensors = discrete_tensors + continuous_tensors + binary_tensors
    all_tensors = [t.unsqueeze(-1) if t.dim() == 1 else t for t in all_tensors]
    combined = torch.cat(all_tensors, dim=1)
    return combined

