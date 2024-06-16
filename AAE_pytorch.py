"""-----------------------------------------------import libraries-----------------------------------------------"""
import main
from synthetic_data import SyntheticData
import argparse
import numpy as np
import pandas as pd
import itertools
import torch
from torch import nn
from torch import Tensor, cuda, exp
from torch.nn import functional as F
from torch.optim import Adam
from torchsummary import summary
from torch.nn import parallel as par
from torch import distributed as dist



"""-----------------------------------------------command-line options-----------------------------------------------"""
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--z_dim", type=int, default=10, help="dimensionality of the latent code")
parser.add_argument("--input_size", type=int, default=32, help="size of each input dimension")
parser.add_argument("--channels", type=int, default=1, help="number of input channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between input sampling")
opt = parser.parse_args()
print(opt)
cuda = True if cuda.is_available() else False

# torch_gpu = torch.empty((20000, 20000)).cuda()
# torch.cuda.memory_allocated()



"""-----------------------------------initialize variables for inputs and outputs-----------------------------------"""
# input_shape_rs = main.X_pca_rs.shape
# input_shape_mas = main.X_pca_mas.shape
df_sel = main.df.iloc[:1000, :100]
in_out_rs = 1000 # in for the enc/gen out for the dec
hl_dim = 150
out_in_dim = 100 # in for the dec and disc out for the enc/gen
z_dim = 10
lr = 0.005



"""-----------------------------------------------------classes-----------------------------------------------------"""
def reparameterization(mu, logvar):
    std = exp(logvar / 2)
    """ 
    stockasticity :
    samples sampled_z from a standard normal distribution (np.random.normal(0, 1, ...)) with the same shape as mu; 
    the mean of the distribution in the latent space.
    Instead of directly using the mean mu to represent the latent variable, the model samples from a distribution around mu.
    
    Backpropagation:
    sampled_z seperate from mu and logvar.
    """
    sampled_z = Tensor(np.random.normal(0, 1, (mu.size(0), opt.z_dim)))
    z = sampled_z * std + mu
    return z


class Residual(nn.Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = nn.Linear(i, o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)



# module compatible with PyTorch
class EncoderGenerator(nn.Module):
    def __init__(self, in_dim, hl_dim, out_dim):
        super(EncoderGenerator, self).__init__()
        seq = []
        for item in list(hl_dim):
            seq += [
                Residual(in_dim, item)
            ]
            in_dim += item

        add_layer = [
            nn.Linear(in_dim, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, out_dim)
        ]
        seq.extend(add_layer)
        self.seq = nn.Sequential(*seq)

        # projects output to the dim of latent space
        self.mu = nn.Linear(out_dim, opt.z_dim)
        self.logvar = nn.Linear(out_dim, opt.z_dim)


# forward propagation
    def forward(self, input_):
        x = self.seq(input_)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self, in_dim, hl_dim, out_dim):
        super(Decoder, self).__init__()
        seq = []
        for item in list(hl_dim):
            seq += [
                Residual(in_dim, item)
            ]
            in_dim += item
        add_layer = [
            nn.Linear(opt.z_dim, in_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, out_dim),
            nn.Tanh(),
        ]
        seq.extend(add_layer)
        self.seq = nn.Sequential(*seq)

    def forward(self, z):
        input_ = self.seq(z)
        return input_



class Discriminator(nn.Module):
    def __init__(self, in_dim, hl_dim, pack=10):
        super(Discriminator, self).__init__()
        dim = in_dim * pack
        self.pack = pack
        self.packdim = dim
        seq = []
        for i in list(hl_dim):
            seq += [
                nn.Linear(opt.z_dim, i),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5),
                nn.Linear(100, 256),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            dim = i
        seq += [nn.Linear(dim, 1),
                nn.Sigmoid()]
        self.seq = nn.Sequential(*seq)

    def forward(self, input_):
        assert input_.size()[0] % self.pack == 0
        return self.seq(input_.view(-1, self.packdim))



def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)



class Cond(object):
    def __init__(self, data, output_info):
        self.model = []

        st = 0
        # skip = False
        max_interval = 0
        counter = 0

        self.interval = []
        self.n_col = 0
        self.n_opt = 0
        st = 0
        self.p = np.zeros((counter, max_interval))
        for i in output_info:
            ed = st + i[0]
            tmp = np.sum(data[:, st:ed], axis=0)
            tmp = np.log(tmp + 1)
            tmp = tmp / np.sum(tmp)
            self.p[self.n_col, :i[0]] = tmp
            self.interval.append((self.n_opt, i[0]))
            self.n_opt += i[0]
            self.n_col += 1
            st = ed
        self.interval = np.asarray(self.interval)

    def sample_rand(self, batch):
        if self.n_col == 0:
            return None
        vec = np.zeros((batch, self.n_opt), dtype='float32')
        idx = np.random.choice(np.arange(self.n_col), batch)
        for i in range(batch):
            col = idx[i]
            pick = int(np.random.choice(self.model[col]))
            vec[i, pick + self.interval[col, 0]] = 1
        return vec





def cond_loss(data, output_info, c, m):
    loss = []
    st = 0
    st_c = 0
    for i in output_info:
        ed = st + i[0]
        ed_c = st_c + i[0]
        tmp = F.cross_entropy(
            data[:, st:ed],
            torch.argmax(c[:, st_c:ed_c], dim=1),
            reduction='none'
        )
        loss.append(tmp)
        st = ed
        st_c = ed_c
    loss = torch.stack(loss, dim=1)

    return (loss * m).sum() / data.size()[0]


class Sampler(object):
    """docstring for Sampler."""

    def __init__(self, data, output_info):
        super(Sampler, self).__init__()
        self.data = data
        self.model = []
        self.n = len(data)

        st = 0
        for item in output_info:
            ed = st + item[0]
            tmp = []
            for j in range(item[0]):
                tmp.append(np.nonzero(data[:, st + j])[0])
            self.model.append(tmp)
            st = ed
        assert st == data.shape[1]

    def sample(self, n, col, opt):
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))
        return self.data[idx]


def calc_gradient_penalty(netD, real_data, fake_data, device='gpu', pac=10, lambda_=10):
    alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
    alpha = alpha.repeat(1, pac, real_data.size(1))
    alpha = alpha.view(-1, real_data.size(1))

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
        create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = (
        (gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1) ** 2).mean() * lambda_
    return gradient_penalty


class Synthesizer(SyntheticData):
    def __init__(self,
                 dataset_name,
                 args=None):

        self.dataset_name = dataset_name
        self.in_dim = args.in_dim
        self.hl_dim = args.hl_dim
        self.lr = args.lr

        self.l2scale = args.l2scale
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.scores_max = 0

    def fit(self, train_data, categorical_columns=tuple(), ordinal_columns=tuple()):

        self.transformer = main.X_pca_rs
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)
        train_data = self.transformer.transform(train_data)

        data_sampler = Sampler(train_data, self.transformer.output_info)
        out_dim = self.transformer.output_dim
        self.cond_enc_gen = Cond(train_data, self.transformer.output_info)

        self.enc_gen = EncoderGenerator(
            self.in_dim + self.cond_enc_gen.n_opt,
            self.hl_dim,
            out_dim)

        self.dec = Decoder(
            out_dim + self.cond_enc_gen.n_opt,
            self.hl_dim,
            self.in_dim)

        discriminator = Discriminator(
            out_dim + self.cond_enc_gen.n_opt,
            self.hl_dim)

        optimizerEG = Adam(
            self.enc_gen.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=self.l2scale)
        # dec
        optimizerD = Adam(discriminator.parameters(), lr=self.lr, betas=(0.5, 0.9))

        if len(train_data) <= self.batch_size:
            self.batch_size = (len(train_data) // 10) * 10

        assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size, self.in_dim, device=self.device)
        std = mean + 1

        steps_per_epoch = len(train_data) // self.batch_size

        for i in range(self.epochs):
            print(i)
            for id_ in range(steps_per_epoch):
                fakez = torch.normal(mean=mean, std=std)

                condvec = self.cond_enc_gen.sample_rand(self.batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = data_sampler.sample(self.batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1)
                    m1 = torch.from_numpy(m1)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c2 = c1[perm]

                fake = self.enc_gen(fakez)

                real = torch.from_numpy(real.astype('float32'))

                if c1 is not None:
                    fake_cat = torch.cat([fake, c1], dim=1)
                    real_cat = torch.cat([real, c2], dim=1)
                else:
                    real_cat = real
                    fake_cat = fake

                y_fake = discriminator(fake_cat)
                y_real = discriminator(real_cat)

                loss_d = -torch.mean(y_real) + torch.mean(y_fake)
                pen = calc_gradient_penalty(discriminator, real_cat, fake_cat)

                optimizerD.zero_grad()
                pen.backward(retain_graph=True)
                loss_d.backward()
                optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self.cond_enc_gen.sample_rand(self.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1)
                    m1 = torch.from_numpy(m1)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.enc_gen(fakez)

                recon_loss = F.mse_loss(fake, fakez)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fake, c1], dim=1))
                else:
                    y_fake = discriminator(fake)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = cond_loss(fake, self.transformer.output_info, c1, m1)

                loss_eg = -torch.mean(y_fake) + recon_loss + cross_entropy

                optimizerEG.zero_grad()
                loss_eg.backward()
                optimizerEG.step()

    def sample(self, n):

        self.enc_gen.eval()

        output_info = self.transformer.output_info
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.in_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std)

            condvec = self.cond_enc_gen.sample_rand(self.batch_size)
            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self.enc_gen(fakez)
            data.append(fake.detach().gpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        return self.transformer.inverse_transform(data, None)






# Use binary cross-entropy loss
# adversarial_loss = nn.BCELoss().cuda() if cuda else nn.BCELoss()
# pixelwise_loss = nn.L1Loss().cuda() if cuda else nn.L1Loss()
# 
# 
# 
# encoder_generator = EncoderGenerator().cuda() if cuda else EncoderGenerator()
# decoder = Decoder().cuda() if cuda else Decoder()
# discriminator = Discriminator().cuda() if cuda else Discriminator()
# 
# 
# 
# optimizer_G = torch.optim.Adam(
#     itertools.chain(encoder_generator.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# 
# def sample_runs(n_row, batches_done):
#     # Sample noise
#     z = Tensor(np.random.normal(0, 1, (n_row ** 2, opt.z_dim)))
#     gen_input = decoder(z)
#     gen_data = gen_input.data.cuda().numpy() if cuda else gen_input.data.numpy()
#     df = pd.DataFrame(gen_data)
#     df.to_csv(f"runs/{batches_done}.csv", index=False)
# 
# 
# data_tensor = torch.tensor(df_sel.values, dtype=torch.float)
# valid = torch.ones((data_tensor.shape[0], 1))
# fake = torch.zeros((data_tensor.shape[0], 1))
# 
# 
# for epoch in range(opt.n_epochs):
#     # Configure input
#     real = data_tensor
#     optimizer_G.zero_grad()
# 
#     encoded = encoder_generator(real)
#     decoded = decoder(encoded)
#     g_loss = 0.001 * adversarial_loss(discriminator(encoded), valid) + 0.999 * pixelwise_loss(
#                 decoded, real
#             )
# 
#     g_loss.backward()
#     optimizer_G.step()
# 
#     # ---------------------
#     #  Train Discriminator
#     # ---------------------
# 
#     optimizer_D.zero_grad()
# 
#     # Sample noise as discriminator ground truth
#     z = Tensor(np.random.normal(0, 1, (data_tensor.shape[0], opt.z_dim)))
# 
#     # Measure discriminator's ability to classify real from generated samples
#     real_loss = adversarial_loss(discriminator(z), valid)
#     fake_loss = adversarial_loss(discriminator(encoded.detach()), fake)
#     d_loss = 0.5 * (real_loss + fake_loss)
# 
#     d_loss.backward()
#     optimizer_D.step()
# 
#     print(
#         epoch, opt.n_epochs, d_loss.item(), g_loss.item()
#     )
# 
#     batches_done = epoch * len(df_sel)
#     if batches_done % opt.sample_interval == 0:
#         sample_runs(n_row=5, batches_done=batches_done)


