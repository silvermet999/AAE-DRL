"""-----------------------------------------------import libraries-----------------------------------------------"""
import os

from sklearn.model_selection import KFold
from torch.nn.utils import prune
from torch.optim.lr_scheduler import MultiStepLR

import math
from AAE import AAE_archi_opt

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy
from torch import cuda
import itertools
from data import main_u
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

cuda = True if cuda.is_available() else False
torch.cuda.empty_cache()
torch.manual_seed(0)
# torch.use_deterministic_algorithms(True)


"""--------------------------------------------------loss and optim--------------------------------------------------"""
encoder_generator = AAE_archi_opt.encoder_generator
decoder = AAE_archi_opt.decoder
discriminator = AAE_archi_opt.discriminator



def l1_regularization(model, rate):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return rate * l1_norm



class MomentumOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.9, noise_decay=0.99):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 < noise_decay <= 1.0:
            raise ValueError("Invalid noise decay value: {}".format(noise_decay))

        defaults = dict(lr=lr, momentum=momentum, noise_decay=noise_decay)
        super(MomentumOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            lr = group['lr']
            noise_decay = group['noise_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # Initialize state if not already done
                if 'momentum_buffer' not in self.state[p]:
                    buf = self.state[p]['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = self.state[p]['momentum_buffer']

                # Apply momentum update
                buf.mul_(momentum).add_(d_p, alpha=lr)

                # Add noise term
                noise = torch.randn_like(p.data) * lr * math.sqrt(1 - noise_decay)
                p.data.add_(-buf + noise)

        return loss



# optimizer_G = torch.optim.Adam(
#     itertools.chain(encoder_generator.parameters(), decoder.parameters()), lr=hyperparams_g["lr"], betas=(hyperparams_g["beta1"], hyperparams_g["beta2"]))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=hyperparams_d["lr"], betas=(hyperparams_d["beta1"], hyperparams_d["beta2"]))
optimizer_G = torch.optim.SGD(itertools.chain(encoder_generator.parameters(), decoder.parameters()), lr=0.000001)
optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=0.0008, weight_decay=0.001)
# optimizer_G = torch.optim.Lookahead(optimizer_G, k=5, alpha=0.5)

# scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 'min', patience=10, factor=0.2, verbose=True)
# scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, 'min', patience=30, factor=0.2, verbose=True)



Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

"""-----------------------------------------------------data gen-----------------------------------------------------"""
def interpolate(z1, z2, n_steps=5):
    interpolations = []
    for alpha in torch.linspace(0, 1, n_steps):
        z = z1 * (1 - alpha) + z2 * alpha
        interpolations.append(z)
    return torch.stack(interpolations)

def sample_runs():
    # z = AAE_archi_opt.custom_dist((batches, z_dim)).cuda() if cuda else AAE_archi_opt.custom_dist((batches, z_dim))
    # gen_input = decoder(z).cpu()
    # gen_data = gen_input.data.numpy()
    # np.savetxt('1.txt', gen_data)
    with torch.no_grad():
        n_interpolations = 4
        n_samples_per_interpolation = 20679
        z1 = torch.randn(n_interpolations, 20).cuda() if cuda else torch.randn(n_interpolations, 16)
        z2 = torch.randn(n_interpolations, 20).cuda() if cuda else torch.randn(n_interpolations, 16)

        # Change: Create a single tensor to store all samples instead of a list
        samples = torch.zeros(n_interpolations * n_samples_per_interpolation, 26).cuda() if cuda else torch.zeros(
            n_interpolations * n_samples_per_interpolation, 26)

        for i in range(n_interpolations):
            interpolations = interpolate(z1[i], z2[i], n_samples_per_interpolation)
            decoded_samples = decoder(interpolations).cuda() if cuda else decoder(interpolations)
            start_idx = i * n_samples_per_interpolation
            end_idx = (i + 1) * n_samples_per_interpolation
            samples[start_idx:end_idx] = decoded_samples

        return samples



"""--------------------------------------------------model training--------------------------------------------------"""
for epoch in range(100):
    for i, (X, y) in enumerate(AAE_archi_opt.dataloader):
        valid = torch.ones((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.ones((X.shape[0], 1),
                                                                                                   requires_grad=False)
        fake = torch.zeros((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.zeros((X.shape[0], 1),
                                                                                                    requires_grad=False)
        real = X.type(Tensor).cuda() if cuda else X.type(Tensor)
        y = y.type(Tensor).unsqueeze(1).cuda() if cuda else y.type(Tensor).unsqueeze(1)
        noisy_real = real + torch.randn_like(real) * 0.4

        discrete_targets = {}
        continuous_targets = {}
        for feature, _ in decoder.discrete_features.items():
            discrete_targets[feature] = torch.ones(noisy_real.shape[0])

        for feature in decoder.continuous_features:
            continuous_targets[feature] = torch.ones(noisy_real.shape[0])

        optimizer_G.zero_grad()
        encoded = encoder_generator(noisy_real)
        dec_input = torch.cat([encoded, y], dim=1)
        discrete_outputs, continuous_outputs = decoder.disc_cont(dec_input)
        l1_pen_g = l1_regularization(decoder, 0.00001)

        g_loss = (0.01 * binary_cross_entropy(discriminator(encoded), valid) +
                  0.99 * decoder.compute_loss((discrete_outputs, continuous_outputs),
                                               (discrete_targets, continuous_targets)) + l1_pen_g)

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        z = AAE_archi_opt.custom_dist((real.shape[0], AAE_archi_opt.z_dim)).cuda() if cuda else AAE_archi_opt.custom_dist((real.shape[0], AAE_archi_opt.z_dim))
        real_loss = binary_cross_entropy(discriminator(z), (valid * (1 - 0.1) + 0.5 * 0.1))
        fake_loss = binary_cross_entropy(discriminator(encoded.detach()), (fake * (1 - 0.1) + 0.5 * 0.1))
        l1_pen_d = l1_regularization(discriminator, 0.0001)
        d_loss = 0.5 * (real_loss + fake_loss) + l1_pen_d

        d_loss.backward()
        optimizer_D.step()

        # scheduler_G.step(g_loss)
        # scheduler_D.step(d_loss)
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


"""--------------------------------------------------model testing---------------------------------------------------"""
recon_losses = []
adversarial_losses = []

for i, (X, y) in enumerate(AAE_archi_opt.val_dl):
    encoder_generator.eval()
    decoder.eval()
    discriminator.eval()

    with torch.no_grad():
        val_real = X.type(Tensor).cuda() if cuda else X.type(Tensor)
        val_target = y.type(Tensor).cuda() if cuda else y.type(Tensor)
        discrete_val = {}
        continuous_val = {}
        for feature, _ in decoder.discrete_features.items():
            discrete_val[feature] = torch.ones(val_real.shape[0])

        for feature in decoder.continuous_features:
            continuous_val[feature] = torch.ones(val_real.shape[0])

        val_encoded = encoder_generator(val_real)
        dec_input = torch.cat([val_encoded, val_target], dim=1)
        val_decoded_disc, val_decoded_cont = decoder.disc_cont(dec_input)

        dec_loss_val = decoder.compute_loss((val_decoded_disc, val_decoded_cont), (discrete_val, continuous_val))


        valid_val = torch.ones((val_real.shape[0], 1)).cuda() if cuda else torch.ones((val_real.shape[0], 1))
        adv_loss_val = binary_cross_entropy(discriminator(val_encoded), valid_val)
# # avg_recon_loss = np.mean(dec_loss_val)
# # avg_adversarial_loss = np.mean(adv_loss_val)
# # print(avg_adversarial_loss)
# # print(avg_recon_loss)
# samples = sample_runs()
# # samples = sample_runs(206134, 28)
# np.savetxt("smp.txt", samples)
# samples = np.concatenate([samples, main_u.X_test_sc])
