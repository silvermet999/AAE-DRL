"""-----------------------------------------------import libraries-----------------------------------------------"""
import os

from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import MultiStepLR

from AAE import AAE_archi

import numpy as np
import torch
from torch.nn import BCELoss, L1Loss
from torch import cuda
from data import main
import itertools
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

cuda = True if cuda.is_available() else False
torch.cuda.empty_cache()


df_train = main.X_train_rs


"""--------------------------------------------------loss and optim--------------------------------------------------"""
adversarial_loss = BCELoss().cuda() if cuda else BCELoss()
recon_loss = L1Loss().cuda() if cuda else L1Loss()


encoder_generator = AAE_archi.encoder_generator
decoder = AAE_archi.decoder
discriminator = AAE_archi.discriminator


def l1_regularization(model, rate):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return rate * l1_norm


hyperparams_g = {'lr': 0.003, 'beta1': 0.86, 'beta2': 0.909}
hyperparams_d = {'lr': 0.0001, 'beta1': 0.78, 'beta2': 0.952}


optimizer_G = torch.optim.Adam(
    itertools.chain(encoder_generator.parameters(), decoder.parameters()), lr=hyperparams_g["lr"], betas=(hyperparams_g["beta1"], hyperparams_g["beta2"]))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=hyperparams_d["lr"], betas=(hyperparams_d["beta1"], hyperparams_d["beta2"]))
# scheduler_D = MultiStepLR(optimizer_D, milestones=[30, 80], gamma=0.01)

# scheduler_G = MultiStepLR(optimizer_G, milestones=[30, 80], gamma=0.01)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



"""-----------------------------------------------------data gen-----------------------------------------------------"""
def interpolate(z1, z2, n_steps=10):
    interpolations = []
    for alpha in torch.linspace(0, 1, n_steps):
        z = z1 * (1 - alpha) + z2 * alpha
        interpolations.append(z)
    return torch.stack(interpolations)

def sample_runs():
    # z = torch.normal(0, 1, (batches, z_dim)).cuda() if cuda else torch.normal(0, 1, (batches, z_dim))
    # z = AAE_archi.custom_dist((batches, z_dim)).cuda() if cuda else AAE_archi.custom_dist((batches, z_dim))
    # gen_input = decoder(z).cpu()
    # gen_data = gen_input.data.numpy()
    # np.savetxt('4.txt', gen_data)
    with torch.no_grad():
        n_interpolations = 4
        n_samples_per_interpolation = 6046
        z1 = torch.randn(n_interpolations, 112).cuda() if cuda else torch.randn(n_interpolations, 112)
        z2 = torch.randn(n_interpolations, 112).cuda() if cuda else torch.randn(n_interpolations, 112)

        samples = []
        for i in range(n_interpolations):
            interpolations = interpolate(z1[i], z2[i], n_samples_per_interpolation)
            decoded_samples = decoder(interpolations).cuda() if cuda else decoder(interpolations)
            samples.append(decoded_samples)

        samples = torch.cat(samples, dim=0)
        return samples


"""--------------------------------------------------model training--------------------------------------------------"""
for epoch in range(200):
    n_batch = len(df_train) // 64
    for i in range(n_batch):
        str_idx = i * 64
        end_idx = str_idx+ 64
        real = torch.tensor(df_train[str_idx:end_idx], dtype=torch.float).cuda() if cuda \
            else torch.tensor(df_train[str_idx:end_idx], dtype=torch.float)
        real = (real - real.mean()) / real.std()
        valid = torch.ones((real.shape[0], 1)).cuda() if cuda else torch.ones((real.shape[0], 1))
        fake = torch.zeros((real.shape[0], 1)).cuda() if cuda else torch.zeros((real.shape[0], 1))
        noisy_real = real + torch.randn_like(real) * 0.1

        optimizer_G.zero_grad()
        # encoded = encoder_generator(real)
        encoded = encoder_generator(noisy_real)
        labels_tensor = AAE_archi.encoded_tensor[str_idx:end_idx]
        dec_input = torch.cat([encoded, labels_tensor], dim=1)
        decoded = decoder(dec_input)
        l1_pen_g = l1_regularization(decoder, 0.00001)
        g_loss = 0.001 * adversarial_loss(discriminator(encoded), valid) + 0.999 * recon_loss(decoded, real) + l1_pen_g

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        # real = (real - real.mean()) / real.std()
        # encoded = encoder_generator(real)
        z = torch.normal(0, 1, (real.shape[0], AAE_archi.z_dim)).cuda() if cuda else torch.normal(0, 1, (real.shape[0], AAE_archi.z_dim))
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded.detach()), fake)
        l1_pen_d = l1_regularization(discriminator, 0.00001)
        d_loss = 0.5 * (real_loss + fake_loss) + l1_pen_d

        d_loss.backward()
        optimizer_D.step()

    # scheduler_G.step()
    # scheduler_D.step()
    print(epoch, d_loss.item(), g_loss.item())

samples = sample_runs()
samples = samples.detach().cpu().numpy()
np.savetxt('5.txt', samples)
print("sample saved")

"""--------------------------------------------------model testing---------------------------------------------------"""
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
recon_losses = []
adversarial_losses = []
for fold, (_, val_index) in enumerate(kf.split(df_train)):
    print(f"Fold {fold + 1}/{n_splits}")
    df_val = df_train[val_index]
    fold_recon_loss = []
    fold_adversarial_loss = []
    encoder_generator.eval()
    decoder.eval()
    discriminator.eval()
    n_batch = len(df_val)
    for i in range(n_batch):
        str_idx = i * 1
        end_idx = str_idx + 1
        with torch.no_grad():
            val_real = torch.tensor(df_val[str_idx:end_idx], dtype=torch.float).cuda() if cuda else torch.tensor(df_val[str_idx:end_idx], dtype=torch.float)
            val_real = (val_real - val_real.mean()) / val_real.std()
            val_encoded = encoder_generator(val_real)
            labels_tensor = AAE_archi.encoded_tensor[str_idx:end_idx]
            dec_input = torch.cat([val_encoded, labels_tensor], dim=1)
            val_decoded = decoder(dec_input)
        recon_loss_val = recon_loss(val_decoded, val_real)
        # val_real = (val_real - val_real.mean()) / val_real.std()
        # val_encoded = encoder_generator(val_real)
        valid_val = torch.ones((val_real.shape[0], 1)).cuda() if cuda else torch.ones((val_real.shape[0], 1))
        adv_loss_val = adversarial_loss(discriminator(val_encoded), valid_val)
        fold_recon_loss.append(recon_loss_val.item())
        fold_adversarial_loss.append(adv_loss_val.item())
    recon_losses.append(np.mean(fold_recon_loss))
    adversarial_losses.append(np.mean(fold_adversarial_loss))
avg_recon_loss = np.mean(recon_losses)
avg_adversarial_loss = np.mean(adversarial_losses)
std_recon_loss = np.std(recon_losses)
std_adversarial_loss = np.std(adversarial_losses)
print(avg_adversarial_loss, std_adversarial_loss)
print(avg_recon_loss, std_recon_loss)
