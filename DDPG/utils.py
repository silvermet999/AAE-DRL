import torch
import numpy as np

from math import ceil, sqrt
import pickle
import os, fnmatch


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


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


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


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
        ind = np.random.randint(len(self.storage))
        self._sample_ind = ind
        return self[ind]

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, items):
        if hasattr(items, '__iter__'):
            items_iter = items
        else:
            items_iter = [items]

        x, y, u, r, d, t = [], [], [], [], [], []
        for i in items_iter:
            X, Y, U, R, D, T = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
            t.append(np.array(T, copy=False))

        return np.array(x).squeeze(0), np.array(y).squeeze(0), np.array(u).squeeze(0), np.array(r).squeeze(0), np.array(d).squeeze(0).reshape(-1, 1), np.array(t).squeeze(0)

    # def save(self, file_dir='./replay'):
    #     if not os.path.exists(os.path.splitext(file_dir)[0]):
    #         os.makedirs(os.path.splitext(file_dir)[0], exist_ok=True)
    #     file_path = os.path.join(os.path.splitext(file_dir)[0], 'replay_{:07}.pkl'.format(len(self)))
    #     with open(file_path, 'wb') as f:
    #         pickle.dump(self.storage[self._ind_to_save:], f, -1)
    #     self._ind_to_save = len(self.storage)
    # 
    # def load(self, file_dir='./replay'):
    #     file_paths = sorted(find('*.pkl', file_dir))
    #     for p in file_paths:
    #         with open(p, 'rb') as f:
    #             self.storage += pickle.load(f)

    # def save_samples(self, sample_num, model_decoder, shuffle=False, save_path='./replay', max_num=10):
    #     """
    #     save some samples from buffer
    #     if shuffle is false indexes 0:sample_num are saved
    # 
    #     Parameters
    #     ----------
    #     sample_num : int
    #         number of samples to save
    #     model_decoder : torch.nn.Module
    #         decoder model
    #     shuffle : bool
    #         get random sample or first elements in buffer
    #     save_path : str
    #         saving folder
    #     max_num : int
    #         maximum number of samples to save
    #     """
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path, exist_ok=True)
    # 
    #     buffer_length = len(self) - 1
    # 
    #     num = min(int(sample_num), max_num)
    #     for i in range(buffer_length, buffer_length - num, -1):
    #         if shuffle:
    #             x, y, u, r, d, t = self.sample(1)
    #             ind = self._sample_ind[0]
    #         else:
    #             x, y, u, r, d, t = self[i]
    #             ind = i
    #         if self._saved[ind]:
    #             return
    #         device = next(model_decoder.parameters()).device
    #         x_tensor = torch.tensor(x).to(device)
    #         y_tensor = torch.tensor(y).to(device)
    #         with torch.no_grad():
    #             x_img = model_decoder(x_tensor)
    #             y_img = model_decoder(y_tensor)
    #         # display_env(x_img, y_img, r, os.path.join(save_path, "img_{}".format(ind + 1)), u)
    #         self._saved[ind] = True



