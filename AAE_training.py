"""-----------------------------------------------import libraries-----------------------------------------------"""
import os

from sklearn.model_selection import KFold
from torch.nn.utils import prune
from torch.optim.lr_scheduler import MultiStepLR


from AAE import AAE_archi_opt

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy
from torch import cuda
import itertools
from data import main_u
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

cuda = False
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

#b increase: rely more on past gradients, slow
# b2 for the update of lr
hyperparams_g = {'lr': 0.002, 'beta1': 0.9, 'beta2': 0.99}
hyperparams_d = {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999}


class MomentumOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9):
        super(MomentumOptimizer, self).__init__(params, defaults={'lr': lr})
        self.momentum = momentum
        self.state = dict()
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(mom=torch.zeros_like(p.data))


    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p not in self.state:
                    self.state[p] = dict(mom=torch.zeros_like(p.data))
                mom = self.state[p]['mom']
                mom = self.momentum * mom - group['lr'] * p.grad.data
                p.data += mom


# optimizer_G = torch.optim.Adam(
#     itertools.chain(encoder_generator.parameters(), decoder.parameters()), lr=hyperparams_g["lr"], betas=(hyperparams_g["beta1"], hyperparams_g["beta2"]))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=hyperparams_d["lr"], betas=(hyperparams_d["beta1"], hyperparams_d["beta2"]))
optimizer_G = torch.optim.SGD(itertools.chain(encoder_generator.parameters(), decoder.parameters(),
                                              ), lr=hyperparams_g["lr"], momentum=0.9)
optimizer_D = torch.optim.SGD(discriminator.parameters(), lr = hyperparams_d["lr"], momentum=0.9)

# scheduler_D = MultiStepLR(optimizer_D, milestones=[20, 80], gamma=0.01)

# scheduler_G = MultiStepLR(optimizer_G, milestones=[20, 80], gamma=0.01)


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
        n_interpolations = 2
        n_samples_per_interpolation = 103067
        z1 = torch.randn(n_interpolations, 28).cuda() if cuda else torch.randn(n_interpolations, 28)
        z2 = torch.randn(n_interpolations, 28).cuda() if cuda else torch.randn(n_interpolations, 28)

        samples = []
        for i in range(n_interpolations):
            interpolations = interpolate(z1[i], z2[i], n_samples_per_interpolation)
            decoded_samples = decoder(interpolations).cuda() if cuda else decoder(interpolations)
            samples.append(decoded_samples)

        samples = torch.cat(samples, dim=0)
    return samples


"""--------------------------------------------------model training--------------------------------------------------"""
for epoch in range(100):
    for i, (X, y) in enumerate(AAE_archi_opt.dataloader):
        valid = torch.ones((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.ones((X.shape[0], 1),
                                                                                                   requires_grad=False)
        fake = torch.zeros((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.zeros((X.shape[0], 1),
                                                                                                    requires_grad=False)
        real = X.type(Tensor).cuda() if cuda else X.type(Tensor)
        y = y.type(Tensor).cuda() if cuda else y.type(Tensor)
        noisy_real = real + torch.randn_like(real) * 0.4

        discrete_targets = {}
        continuous_targets = {}
        current_idx = 0
        for feature, num_classes in decoder.discrete_features.items():
            feature_data = noisy_real[:, current_idx:current_idx + num_classes]
            discrete_targets[feature] = feature_data
            current_idx += num_classes

        for feature in decoder.continuous_features:
            feature_data = noisy_real[:, current_idx:current_idx + 1]
            continuous_targets[feature] = feature_data
            current_idx += 1

        optimizer_G.zero_grad()
        encoded = encoder_generator(noisy_real)
        dec_input = torch.cat([encoded, y], dim=1)
        discrete_outputs, continuous_outputs = decoder(dec_input)
        l1_pen_g = l1_regularization(decoder, 0.00001)
        g_loss = 0.001 * binary_cross_entropy(discriminator(encoded), valid) + 0.999 * decoder.compute_loss((discrete_outputs, continuous_outputs), (discrete_targets, continuous_targets)) + l1_pen_g

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        z = AAE_archi_opt.custom_dist((real.shape[0], AAE_archi_opt.z_dim)).cuda() if cuda else AAE_archi_opt.custom_dist((real.shape[0], AAE_archi_opt.z_dim))
        real_loss = binary_cross_entropy(discriminator(z), valid)
        fake_loss = binary_cross_entropy(discriminator(encoded.detach()), fake)
        l1_pen_d = l1_regularization(discriminator, 0.0001)
        d_loss = 0.5 * (real_loss + fake_loss) + l1_pen_d

        d_loss.backward()
        optimizer_D.step()

        # scheduler_G.step()
        # scheduler_D.step()
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
        val_encoded = encoder_generator(val_real)
        dec_input = torch.cat([val_encoded, y.type(Tensor).cuda() if cuda else y.type(Tensor)], dim=1)
        val_decoded = decoder(dec_input)
    recon_loss_val = cross_entropy(val_decoded, val_real)
    val_encoded = encoder_generator(val_real)
    valid_val = torch.ones((val_real.shape[0], 1)).cuda() if cuda else torch.ones((val_real.shape[0], 1))
    adv_loss_val = binary_cross_entropy(discriminator(val_encoded), valid_val)
avg_recon_loss = np.mean(recon_loss_val.item())
avg_adversarial_loss = np.mean(adv_loss_val.item())
print(avg_adversarial_loss)
print(avg_recon_loss)
samples = sample_runs()
samples = samples.detach().cpu()
# samples = sample_runs(206134, 28)
np.savetxt("2.txt", samples)
# samples = np.concatenate([samples, main_u.X_test_sc])
# np.savetxt("3.txt", samples)
