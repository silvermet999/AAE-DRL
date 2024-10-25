"""-----------------------------------------------import libraries-----------------------------------------------"""
import os

from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import MultiStepLR

from AAE import AAE_archi

import numpy as np
import torch
from torch.nn import BCELoss, L1Loss
from torch import cuda
from torch.utils.data import DataLoader, Dataset
from data import main
import itertools
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

cuda = True if cuda.is_available() else False
torch.cuda.empty_cache()



class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

X_train = main.X_train_rs[:59280]
y_train = main.y_train_np[:59280]
X_val = main.X_train_rs[-14820:]
y_val = main.y_train_np[-14820:]
dataset = CustomDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
val = CustomDataset(X_val, y_val)
val_dl = DataLoader(dataset, batch_size=1)

"""--------------------------------------------------loss and optim--------------------------------------------------"""
adversarial_loss = BCELoss().cuda() if cuda else BCELoss()
recon_loss = L1Loss().cuda() if cuda else L1Loss()


encoder_generator = AAE_archi.encoder_generator
decoder = AAE_archi.decoder
discriminator = AAE_archi.discriminator


def l1_regularization(model, rate):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return rate * l1_norm


hyperparams_g = {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.9}
hyperparams_d = {'lr': 0.00002, 'beta1': 0.9, 'beta2': 0.95}


optimizer_G = torch.optim.Adam(
    itertools.chain(encoder_generator.parameters(), decoder.parameters()), lr=hyperparams_g["lr"], betas=(hyperparams_g["beta1"], hyperparams_g["beta2"]))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=hyperparams_d["lr"], betas=(hyperparams_d["beta1"], hyperparams_d["beta2"]))
scheduler_D = MultiStepLR(optimizer_D, milestones=[20, 80], gamma=0.01)

scheduler_G = MultiStepLR(optimizer_G, milestones=[20, 80], gamma=0.01)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# class PruningScheduler:
#     def __init__(self, model, pruning_steps, pruning_amount):
#         self.model = model
#         self.pruning_steps = pruning_steps
#         self.pruning_amount = pruning_amount
#         self.current_step = 0
#
#     def step(self):
#         if self.current_step % self.pruning_steps == 0:
#             print(f'Pruning at step {self.current_step}')
#             prune.l1_unstructured(self.model.seq, name='weight', amount=self.pruning_amount)
#         self.current_step += 1
#
# pruning_scheduler = PruningScheduler(encoder_generator, pruning_steps=100, pruning_amount=0.1)

"""-----------------------------------------------------data gen-----------------------------------------------------"""
def interpolate(z1, z2, n_steps=10):
    interpolations = []
    for alpha in torch.linspace(0, 1, n_steps):
        z = z1 * (1 - alpha) + z2 * alpha
        interpolations.append(z)
    return torch.stack(interpolations)

def sample_runs():
    # z = torch.normal(0, 1, (batches, z_dim)).cuda() if cuda else torch.normal(0, 1, (batches, z_dim))
    # # z = AAE_archi.custom_dist((batches, z_dim)).cuda() if cuda else AAE_archi.custom_dist((batches, z_dim))
    # gen_input = decoder(z).cpu()
    # gen_data = gen_input.data.numpy()
    # np.savetxt('1.txt', gen_data)
    with torch.no_grad():
        n_interpolations = 4
        n_samples_per_interpolation = 18525
        z1 = torch.randn(n_interpolations, 111).cuda() if cuda else torch.randn(n_interpolations, 111)
        z2 = torch.randn(n_interpolations, 111).cuda() if cuda else torch.randn(n_interpolations, 111)

        samples = []
        for i in range(n_interpolations):
            interpolations = interpolate(z1[i], z2[i], n_samples_per_interpolation)
            decoded_samples = decoder(interpolations).cuda() if cuda else decoder(interpolations)
            samples.append(decoded_samples)

        samples = torch.cat(samples, dim=0)
        return samples


"""--------------------------------------------------model training--------------------------------------------------"""
for epoch in range(100):
    if epoch % 20 == 0:
        AAE_archi.apply_pruning_to_model(encoder_generator)
    for i, (X, y) in enumerate(dataloader):
        valid = torch.ones((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.ones((X.shape[0], 1),
                                                                                                   requires_grad=False)
        fake = torch.zeros((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.zeros((X.shape[0], 1),
                                                                                                    requires_grad=False)
        real = X.type(Tensor).cuda() if cuda else X.type(Tensor)
        y = y.type(Tensor).cuda() if cuda else y.type(Tensor)
        real = (real - real.mean()) / real.std()
        noisy_real = real + torch.randn_like(real) * 0.2

        optimizer_G.zero_grad()
        encoded = encoder_generator(noisy_real)
        dec_input = torch.cat([encoded, y], dim=1)
        decoded = decoder(dec_input)
        l1_pen_g = l1_regularization(decoder, 0.00001)
        g_loss = 0.001 * adversarial_loss(discriminator(encoded), valid) + 0.999 * recon_loss(decoded, noisy_real) + l1_pen_g

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        z = torch.normal(0, 1, (real.shape[0], AAE_archi.z_dim)).cuda() if cuda else torch.normal(0, 1, (real.shape[0], AAE_archi.z_dim))
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded.detach()), fake)
        l1_pen_d = l1_regularization(discriminator, 0.00001)
        d_loss = 0.5 * (real_loss + fake_loss) + l1_pen_d

        d_loss.backward()
        optimizer_D.step()
        # pruning_scheduler.step()
    # scheduler_G.step()
    # scheduler_D.step()
    print(epoch, d_loss.item(), g_loss.item())

# samples = sample_runs(74100, 111)
samples = sample_runs()
samples = samples.detach().cpu()
np.savetxt("3.txt", samples)
print("sample saved")

"""--------------------------------------------------model testing---------------------------------------------------"""
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
recon_losses = []
adversarial_losses = []
for fold, (X, y) in enumerate(kf.split(val_dl)):
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
