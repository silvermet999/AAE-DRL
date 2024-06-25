"""-----------------------------------------------import libraries-----------------------------------------------"""
import main
import clf
from synthetic_data import SyntheticData
import argparse
import numpy as np
import pandas as pd
import itertools
from torch import cat, rand, autograd, ones, FloatTensor, tensor, zeros, float
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.nn import BatchNorm1d, LeakyReLU, Linear, Module, Sequential, Tanh, Sigmoid, BCELoss, L1Loss
from torch import Tensor, cuda, exp
from torchsummary import summary
import mlflow



"""-----------------------------------------------command-line options-----------------------------------------------"""
cuda = True if cuda.is_available() else False

# torch_gpu = torch.empty((20000, 20000)).cuda()
# torch.cuda.memory_allocated()



"""-----------------------------------initialize variables for inputs and outputs-----------------------------------"""
df = pd.DataFrame(main.X_rs, columns=main.X.columns)
df_sel = df.iloc[:5000]
in_out_rs = 127 # in for the enc/gen out for the dec
hl_dim = (100, 100, 100, 100, 100)
hl_dimd = (10, 10, 10, 10, 10)
out_in_dim = 100 # in for the dec and disc out for the enc/gen
z_dim = 10
params = {
    "lr": 0.01,
    "batch_size": 24,
    "n_epochs": 100,
    "optimizer": "Adam",
    "gamma": 0.9
}



"""--------------------------------------------backprop and hidden layers--------------------------------------------"""
def reparameterization(mu, logvar, z_dim):
    std = exp(logvar / 2)
    """ 
    stockasticity :
    samples sampled_z from a standard lognormal distribution (np.random.lognormal(0, 1, ...)) with the same shape as mu; 
    the mean of the distribution in the latent space.
    Instead of directly using the mean mu to represent the latent variable, the model samples from a distribution around mu.
    
    Backpropagation:
    sampled_z seperate from mu and logvar.
    """
    sampled_z = Tensor(np.random.lognormal(0, 1, (mu.size(0), z_dim)))
    z = sampled_z * std + mu
    return z


#loop for hidden layers
class Residual(Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc1 = Linear(i, o)
        self.leakyrelu1 = LeakyReLU(0.2)
        self.bn = BatchNorm1d(o)

    def forward(self, l0):
        l1 = self.fc1(l0)
        l2 = self.leakyrelu1(l1)
        l3= self.bn(l2)
        return cat([l3, l0], dim=1)



"""----------------------------------------------------AAE blocks----------------------------------------------------"""
# module compatible with PyTorch
class EncoderGenerator(Module):
    def __init__(self):
        super(EncoderGenerator, self).__init__()
        dim = in_out_rs
        seq = []
        for i in list(hl_dim):
            seq += [Residual(dim, i)]
            dim += i
        seq.append(Linear(dim, out_in_dim))
        self.seq = Sequential(*seq)


        # projects output to the dim of latent space
        self.mu = Linear(out_in_dim, z_dim)
        self.logvar = Linear(out_in_dim, z_dim)

# forward propagation
    def forward(self, input_):
        x = self.seq(input_)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar, z_dim)
        return z


class Decoder(Module):
    def __init__(self):
        super(Decoder, self).__init__()
        dim = z_dim
        seq = []
        for i in list(hl_dim):
            seq += [Residual(dim, i)]
            dim += i
        seq += [Linear(dim, in_out_rs), Tanh()]
        self.seq = Sequential(*seq)

    def forward(self, z):
        input_ = self.seq(z)
        return input_


class Discriminator(Module):
    def __init__(self, pack=10):
        super(Discriminator, self).__init__()
        dim = z_dim * pack
        self.pack = pack
        self.packdim = dim
        seq = []
        for i in list(hl_dimd):
            seq += [
                Linear(z_dim, i),
                LeakyReLU(0.2, inplace=True),
                Linear(10, 10),
                LeakyReLU(0.2, inplace=True),
            ]
            dim = i
        seq += [Linear(dim, 1), Sigmoid()]
        self.seq = Sequential(*seq)

    def forward(self, input_):
        return self.seq(input_)



"""--------------------------------------------------loss and optim--------------------------------------------------"""
adversarial_loss = BCELoss().cuda() if cuda else BCELoss()
recon_loss = L1Loss().cuda() if cuda else L1Loss()


encoder_generator = EncoderGenerator().cuda() if cuda else (
    EncoderGenerator())
summary(encoder_generator, input_size=(in_out_rs,))
decoder = Decoder().cuda() if cuda else Decoder()
summary(decoder, input_size=(z_dim,))
discriminator = Discriminator().cuda() if cuda else Discriminator()
summary(discriminator, input_size=(z_dim,))


def calc_gradient_penalty(netD, real_data, fake_data, device='cpu', pac=10, lambda_=10):
    alpha = rand(real_data.size(0) // pac, 1, 1, device=device)
    alpha = alpha.repeat(1, pac, real_data.size(1))
    alpha = alpha.view(-1, real_data.size(1))

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=ones(disc_interpolates.size(), device=device),
        create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = (
        (gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1) ** 2).mean() * lambda_
    return gradient_penalty


mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("test_experiment")
mlflow.log_param(params)


optimizer_G = Adam(
    itertools.chain(encoder_generator.parameters(), decoder.parameters()), lr=0.01, betas=(0.5, 0.99))
optimizer_D = Adam(discriminator.parameters(), lr=0.01, betas=(0.01, 0.9))
scheduler1 = MultiStepLR(optimizer_G, milestones=[30,80], gamma=0.1)
scheduler2 = ExponentialLR(optimizer_D, gamma=0.9)



"""-----------------------------------------------------data gen-----------------------------------------------------"""
def sample_runs(n_row, z_dim, batches_done):
    # Sample noise
    z = Tensor(np.random.lognormal(0, 1, (n_row ** 2, z_dim)))
    gen_input = decoder(z)
    gen_data = gen_input.data.cuda().numpy() if cuda else gen_input.data.numpy()
    df = pd.DataFrame(gen_data)
    df.to_csv(f"runs/{batches_done}.csv", index=False)



"""--------------------------------------------------model training--------------------------------------------------"""
for epoch in range(100):
    n_batch = len(df_sel) // 24
    for i in range(n_batch):
        str_idx = i * 24
        end_idx = str_idx + 24
        batch_data = df_sel.iloc[str_idx:end_idx]
        batch_tensor = tensor(batch_data.values, dtype=float).cuda() if cuda else tensor(
            batch_data.values, dtype=float)

        real = (batch_tensor - batch_tensor.mean()) / batch_tensor.std()
        valid = ones((batch_tensor.shape[0], 1)).cuda() if cuda else ones((batch_tensor.shape[0], 1))
        fake = zeros((batch_tensor.shape[0], 1)).cuda() if cuda else zeros((batch_tensor.shape[0], 1))
        optimizer_G.zero_grad()

        encoded = encoder_generator(real)
        decoded = decoder(encoded)

        g_loss = 0.001 * adversarial_loss(discriminator(encoded), valid) + 0.999 * recon_loss(
                    decoded, real
                )
        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        z = Tensor(np.random.lognormal(0, 1, (batch_tensor.shape[0], z_dim)))
        # real and fake loss should be close
        # discriminator(z) should be close to 0
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)
        d_loss.backward()
        optimizer_D.step()

        scheduler1.step()
        scheduler2.step()

        print(epoch, n_batch, d_loss.item(), g_loss.item())
        batches_done = epoch * len(df_sel)
        if batches_done % 400 == 0:
            sample_runs(n_row=71, z_dim=10, batches_done=3)


    print(clf.xgb())


# mlflow server --host 127.0.0.1 --port 8080



