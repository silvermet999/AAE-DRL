"""-----------------------------------------------import libraries-----------------------------------------------"""
import os

from sklearn.model_selection import KFold
from torch.nn.utils import prune
from torch.optim.lr_scheduler import MultiStepLR


from AAE import AAE_archi_opt

import numpy as np
import torch
from torch.nn import BCELoss, L1Loss
from torch import cuda
import itertools
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

cuda = True if cuda.is_available() else False
torch.cuda.empty_cache()





"""--------------------------------------------------loss and optim--------------------------------------------------"""
adversarial_loss = BCELoss().cuda() if cuda else BCELoss()
recon_loss = L1Loss().cuda() if cuda else L1Loss()


encoder_generator = AAE_archi_opt.encoder_generator
decoder = AAE_archi_opt.decoder
discriminator = AAE_archi_opt.discriminator


def l1_regularization(model, rate):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return rate * l1_norm

#b increase: rely more on past gradients, slow
# b2 for the update of lr
hyperparams_g = {'lr': 0.001, 'beta1': 0.99, 'beta2': 0.99}
hyperparams_d = {'lr': 0.001, 'beta1': 0.6, 'beta2': 0.6}
# hyperparams_g = {'lr': 0.00005, 'beta1': 0.7, 'beta2': 0.96}
# hyperparams_d = {'lr': 0.002, 'beta1': 0.55, 'beta2': 0.94}


# optimizer_G = torch.optim.Adam(
#     itertools.chain(encoder_generator.parameters(), decoder.parameters()), lr=hyperparams_g["lr"], betas=(hyperparams_g["beta1"], hyperparams_g["beta2"]))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=hyperparams_d["lr"], betas=(hyperparams_d["beta1"], hyperparams_d["beta2"]))
optimizer_G = torch.optim.SGD(itertools.chain(encoder_generator.parameters(), decoder.parameters(),
                                              ), lr=hyperparams_g["lr"], momentum=0.4)
optimizer_D = torch.optim.SGD(discriminator.parameters(), lr = hyperparams_d["lr"], momentum=0.99)

scheduler_D = MultiStepLR(optimizer_D, milestones=[20, 80], gamma=0.01)

scheduler_G = MultiStepLR(optimizer_G, milestones=[20, 80], gamma=0.01)


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

"""-----------------------------------------------------data gen-----------------------------------------------------"""
def interpolate(z1, z2, n_steps=5):
    interpolations = []
    for alpha in torch.linspace(0, 1, n_steps):
        z = z1 * (1 - alpha) + z2 * alpha
        interpolations.append(z)
    return torch.stack(interpolations)

def sample_runs():
    # z = torch.normal(0, 1, (batches, z_dim)).cuda() if cuda else torch.normal(0, 1, (batches, z_dim))
    # # # z = AAE_archi_opt.custom_dist((batches, z_dim)).cuda() if cuda else AAE_archi_opt.custom_dist((batches, z_dim))
    # gen_input = decoder(z).cpu()
    # gen_data = gen_input.data.numpy()
    # np.savetxt('1.txt', gen_data)
    with torch.no_grad():
        n_interpolations = 2
        n_samples_per_interpolation = 103067
        z1 = torch.randn(n_interpolations, 22).cuda() if cuda else torch.randn(n_interpolations, 22)
        z2 = torch.randn(n_interpolations, 22).cuda() if cuda else torch.randn(n_interpolations, 22)

        samples = []
        for i in range(n_interpolations):
            interpolations = interpolate(z1[i], z2[i], n_samples_per_interpolation)
            decoded_samples = decoder(interpolations).cuda() if cuda else decoder(interpolations)
            samples.append(decoded_samples)

        samples = torch.cat(samples, dim=0)
        return samples


"""--------------------------------------------------model training--------------------------------------------------"""
for epoch in range(200):
    for i, (X, y) in enumerate(AAE_archi_opt.dataloader):
        valid = torch.ones((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.ones((X.shape[0], 1),
                                                                                                   requires_grad=False)
        fake = torch.zeros((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.zeros((X.shape[0], 1),
                                                                                                    requires_grad=False)
        real = X.type(Tensor).cuda() if cuda else X.type(Tensor)
        y = y.type(Tensor).cuda() if cuda else y.type(Tensor)
        # real = (real - real.mean()) / real.std()
        # noisy_real = real + torch.randn_like(real) * 0.2

        optimizer_G.zero_grad()
        encoded = encoder_generator(real)
        dec_input = torch.cat([encoded, y], dim=1)
        decoded = decoder(dec_input)
        l1_pen_g = l1_regularization(decoder, 0.00001)
        g_loss = 0.001 * adversarial_loss(discriminator(encoded), valid) + 0.999 * recon_loss(decoded, real) + l1_pen_g

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        real = (real - real.mean()) / real.std()
        z = torch.normal(0, 1, (real.shape[0], AAE_archi_opt.z_dim)).cuda() if cuda else torch.normal(0, 1, (real.shape[0], AAE_archi_opt.z_dim))
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded.detach()), fake)
        l1_pen_d = l1_regularization(discriminator, 0.0001)
        d_loss = 0.5 * (real_loss + fake_loss) + l1_pen_d

        d_loss.backward()
        optimizer_D.step()

        scheduler_G.step()
        scheduler_D.step()
    # if epoch == 20:
    #     print("apply")
    #     AAE_archi_opt.apply_pruning(encoder_generator)
    # if epoch == 60:
    #     print("remove")
    #     AAE_archi_opt.remove_pruning(encoder_generator)

    # d_output_real = real_loss.mean().item()
    # d_output_fake = fake_loss.mean().item()
    # if d_output_real < 0.5 or d_output_fake > 0.5:
    #     for param_group in optimizer_G.param_groups:
    #         param_group['lr'] *= 1.05
    # else:
    #     for param_group in optimizer_G.param_groups:
    #         param_group['lr'] *= 0.95

    print(epoch, d_loss.item(), g_loss.item())


    if epoch == 199:
        samples = sample_runs()
        samples = samples.detach().cpu()
        np.savetxt("3.txt", samples)
        print("sample saved")

# samples = sample_runs(16893, 46)
# print("sample saved")

"""--------------------------------------------------model testing---------------------------------------------------"""
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
recon_losses = []
adversarial_losses = []
for fold, (X, y) in enumerate(kf.split(AAE_archi_opt.val_dl)):
    print(f"Fold {fold + 1}/{n_splits}")
    fold_recon_loss = []
    fold_adversarial_loss = []
    encoder_generator.eval()
    decoder.eval()
    discriminator.eval()
    with torch.no_grad():
        val_real = X.type(Tensor).cuda() if cuda else X.type(Tensor)
        val_real = (val_real - val_real.mean()) / val_real.std()
        val_encoded = encoder_generator(val_real)
        dec_input = torch.cat([val_encoded, y], dim=1)
        val_decoded = decoder(dec_input)
    recon_loss_val = recon_loss(val_decoded, val_real)
    val_encoded = encoder_generator(val_real)
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
