"""-----------------------------------------------import libraries-----------------------------------------------"""
from torch.optim.lr_scheduler import MultiStepLR

import dim_reduction
import argparse
import numpy as np
import pandas as pd
import itertools
import torch
from torch.nn import BatchNorm1d, LeakyReLU, Linear, Module, Sequential, Tanh, Sigmoid, BCELoss, L1Loss
from torch import Tensor, cuda, exp
from torchsummary import summary
import mlflow



"""-----------------------------------------------command-line options-----------------------------------------------"""
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
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
df_sel = dim_reduction.x_pca_train[:5000]
in_out_rs = 100 # in for the enc/gen out for the dec
hl_dim = (100, 100, 100, 100, 100)
hl_dimd = (10, 10, 10, 10, 10)
out_in_dim = 40 # in for the dec and disc out for the enc/gen
z_dim = 10
params = {
    "lr": 0.01,
    "batch_size": 24,
    "n_epochs": 100,
    "optimizer": "Adam",
    "gamma": 0.9
}



"""---------------------------------------------backprop and hidden layers-------------------------------------------"""
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



"""----------------------------------------------------AAE blocks----------------------------------------------------"""
# module compatible with PyTorch
class EncoderGenerator(Module):
    def __init__(self):
        super(EncoderGenerator, self).__init__()
        dim = in_out_rs
        seq = []
        for i in list(hl_dim):
            seq += [hl_loop(dim, i)]
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
            seq += [hl_loop(dim, i)]
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
# Use binary cross-entropy loss
adversarial_loss = BCELoss().cuda() if cuda else BCELoss()
recon_loss = L1Loss().cuda() if cuda else L1Loss()


encoder_generator = EncoderGenerator().cuda() if cuda else (
    EncoderGenerator())
summary(encoder_generator, input_size=(in_out_rs,))
decoder = Decoder().cuda() if cuda else Decoder()
summary(decoder, input_size=(z_dim,))
discriminator = Discriminator().cuda() if cuda else Discriminator()
summary(discriminator, input_size=(z_dim,))


optimizer_G = torch.optim.Adam(
    itertools.chain(encoder_generator.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
scheduler_D = MultiStepLR(optimizer_G, milestones=[30,80], gamma=0.1)
scheduler_G = MultiStepLR(optimizer_G, milestones=[30,80], gamma=0.1)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("test_experiment")

"""-----------------------------------------------------data gen-----------------------------------------------------"""
def sample_runs(n_row, z_dim, batches_done):
    z = Tensor(np.random.lognormal(0, 1, (n_row ** 2, z_dim)))
    gen_input = decoder(z)
    gen_data = gen_input.data.cuda().numpy() if cuda else gen_input.data.numpy()
    dim_reduction.x_pca_train = pd.DataFrame(gen_data)
    dim_reduction.x_pca_train.to_csv(f"runs/{batches_done}.csv", index=False)



"""--------------------------------------------------model training--------------------------------------------------"""
for epoch in range(100):
    n_batch = len(df_sel) // 24
    for i in range(n_batch):
        str_idx = i * 24
        end_idx = str_idx + 24
        batch_data = df_sel[str_idx:end_idx]
        batch_tensor = torch.tensor(batch_data, dtype=torch.float).cuda() if cuda else torch.tensor(
            batch_data, dtype=torch.float)

        real = (batch_tensor - batch_tensor.mean()) / batch_tensor.std()
        valid = torch.ones((batch_tensor.shape[0], 1)).cuda() if cuda else torch.ones((batch_tensor.shape[0], 1))
        fake = torch.zeros((batch_tensor.shape[0], 1)).cuda() if cuda else torch.zeros((batch_tensor.shape[0], 1))
        optimizer_G.zero_grad()

        encoded = encoder_generator(real)
        decoded = decoder(encoded)
        g_loss = 0.001 * adversarial_loss(discriminator(encoded), valid) + 0.999 * recon_loss(
                    decoded, real
                )

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = Tensor(np.random.lognormal(0, 1, (batch_tensor.shape[0], z_dim)))

        # real and fake loss should be close
        # discriminator(z) should be close to 0
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

    scheduler_G.step()
    scheduler_D.step()
    print(epoch, opt.n_epochs, d_loss.item(), g_loss.item())

    batches_done = epoch * len(df_sel)
    if batches_done % opt.sample_interval == 0:
        sample_runs(n_row=71, z_dim=10, batches_done=3)

    # print(clf.xgb())


with mlflow.start_run():
    mlflow.log_params(params)

    mlflow.log_metric("loss", g_loss)
    mlflow.log_metric("d_loss", d_loss)

    mlflow.set_tag("Training Info", "Test")

    model_info_gen = mlflow.sklearn.log_model(
        sk_model = encoder_generator,
        artifact_path="mlflow/gen",
        input_example=in_out_rs,
        registered_model_name="G_tracking",
    )
    model_info_disc = mlflow.sklearn.log_model(
        sk_model=discriminator,
        artifact_path="mlflow/discriminator",
        input_example=z_dim,
        registered_model_name="D_tracking",
    )


