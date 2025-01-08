"""-----------------------------------------------import libraries-----------------------------------------------"""
import os
import pandas as pd
from sklearn.model_selection import KFold
from torch.nn.utils import prune
from torch.optim.lr_scheduler import MultiStepLR

from AAE import AAE_archi_opt
import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy, one_hot
from torch import cuda
import itertools
from data import main_u
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

cuda = True if torch.cuda.is_available() else False
torch.cuda.empty_cache()
torch.manual_seed(0)
# {'mean_g_loss': 0.3462480198987731, 'mean_d_loss': 0.3994785000063839, 'std_g_loss': 0.019267043659791275, 'std_d_loss': 0.108150935979322, 'fold_metrics': [{'avg_g_loss': 0.31882497365499834, 'avg_d_loss': 0.2079493176529308}, {'avg_g_loss': 0.3315604346634534, 'avg_d_loss': 0.3700757668715135}, {'avg_g_loss': 0.35120319234593816, 'avg_d_loss': 0.4995852775649222}, {'avg_g_loss': 0.35551101834801613, 'avg_d_loss': 0.5024025525151605}, {'avg_g_loss': 0.3741404804814593, 'avg_d_loss': 0.4173795854273926}]}

"""--------------------------------------------------loss and optim--------------------------------------------------"""
encoder_generator = AAE_archi_opt.encoder_generator
decoder = AAE_archi_opt.decoder
discriminator = AAE_archi_opt.discriminator


def l2_reg(model, coef):
    reg_loss = 0
    for param in model.parameters():
        reg_loss += coef * torch.norm(param) ** 2
    return reg_loss

def l1_reg(model, rate):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return rate * l1_norm


criterion = torch.nn.CrossEntropyLoss()


optimizer_G = torch.optim.SGD(itertools.chain(encoder_generator.parameters(), decoder.parameters()), lr=0.001)
optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=0.001)

# scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 'min', patience=10, factor=0.2, verbose=True)


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

"""-----------------------------------------------------data gen-----------------------------------------------------"""
def save_features_to_csv(discrete_samples, continuous_samples, binary_samples):
    def dict_to_df(tensor_dict):
        all_data = []
        for sample_idx in range(next(iter(tensor_dict.values())).shape[0]):
            row_data = {}
            for feature_name, tensor in tensor_dict.items():
                if len(tensor.shape) > 2:
                    tensor = tensor.reshape(tensor.shape[0], -1)

                values = tensor[sample_idx].detach().cpu().numpy()
                if len(values.shape) == 0:
                    row_data[f"{feature_name}"] = values.item()
                else:
                    for _, value in enumerate(values):
                        row_data[f"{feature_name}"] = value
            all_data.append(row_data)
        return pd.DataFrame(all_data)

    discrete_df = dict_to_df(discrete_samples)
    continuous_df = dict_to_df(continuous_samples)
    binary_df = dict_to_df(binary_samples)

    combined_df = pd.concat([discrete_df, continuous_df, binary_df], axis=1)
    combined_df.to_csv('all_features.csv')

    return combined_df


def interpolate(z1, z2, n_steps=5):
    interpolations = []
    for alpha in torch.linspace(0, 1, n_steps):
        z = z1 * (1 - alpha) + z2 * alpha
        interpolations.append(z)
    return torch.stack(interpolations)

def sample_runs():
    discrete_samples = {feature: [] for feature in decoder.discrete_features}
    continuous_samples = {feature: [] for feature in decoder.continuous_features}
    binary_samples = {feature: [] for feature in decoder.binary_features}
    with torch.no_grad():
        n_interpolations = 4
        n_samples_per_interpolation = 90711
        z1 = torch.randn(n_interpolations, 10).cuda() if cuda else torch.randn(n_interpolations, 10)
        z2 = torch.randn(n_interpolations, 10).cuda() if cuda else torch.randn(n_interpolations, 10)

        for i in range(n_interpolations):
            interpolations = interpolate(z1[i], z2[i], n_samples_per_interpolation)
            discrete_out, continuous_out, binary_out = decoder.disc_cont(interpolations).cuda() if cuda else decoder.disc_cont(
                interpolations)

            for feature in decoder.discrete_features:
                discrete_samples[feature].append(torch.argmax(torch.round(discrete_out[feature]), dim=-1))

            for feature in decoder.continuous_features:
                continuous_samples[feature].append(continuous_out[feature])

            for feature in decoder.binary_features:
                binary_samples[feature].append(torch.argmax(torch.round(binary_out[feature]), dim=-1))


        for feature in decoder.discrete_features:
            discrete_samples[feature] = torch.cat(discrete_samples[feature], dim=0)

        for feature in decoder.continuous_features:
            continuous_samples[feature] = torch.cat(continuous_samples[feature], dim=0)

        for feature in decoder.binary_features:
            binary_samples[feature] = torch.cat(binary_samples[feature], dim=0)
        return discrete_samples, continuous_samples, binary_samples


"""-------------------------------------------------------KFold------------------------------------------------------"""

def get_kfold_indices(dataset, n_splits=5, shuffle=True, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    indices = [(train_indices, val_indices) for train_indices, val_indices in kf.split(dataset)]
    return indices

def kfold_cross_validation():
    kfold_indices = get_kfold_indices(AAE_archi_opt.dataset)
    fold_metrics = []
    for fold, indices in enumerate(kfold_indices, 1):
        train_loader, val_loader = AAE_archi_opt.dataset_function(AAE_archi_opt.dataset, train=True)

        train_model(train_loader)

        fold_metrics.append(evaluate_model(val_loader))
        print(f"Fold {fold} completed. Metrics: {fold_metrics[-1]}")

    torch.save(encoder_generator.state_dict(), 'enc.pth')
    torch.save(decoder.state_dict(), "dec.pth")
    torch.save(discriminator.state_dict(), "disc.pth")




    cv_results = {
        'mean_g_loss': np.mean([metrics['avg_g_loss'] for metrics in fold_metrics]),
        'mean_d_loss': np.mean([metrics['avg_d_loss'] for metrics in fold_metrics]),

        'std_g_loss': np.std([metrics['avg_g_loss'] for metrics in fold_metrics]),
        'std_d_loss': np.std([metrics['avg_d_loss'] for metrics in fold_metrics]),
        'fold_metrics': fold_metrics
    }
    return cv_results


"""--------------------------------------------------model training--------------------------------------------------"""
def train_model(train_loader):
    for epoch in range(10):
        for i, (X, y) in enumerate(train_loader):
            valid = torch.ones((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.ones((X.shape[0], 1),
                                                                                                       requires_grad=False)
            fake = torch.zeros((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.zeros((X.shape[0], 1),
                                                                                                        requires_grad=False)
            real = X.type(Tensor).cuda() if cuda else X.type(Tensor)
            y = y.type(Tensor).cuda() if cuda else y.type(Tensor)
            noisy_real = real + torch.randn_like(real) * 0.3

            discrete_targets = {}
            continuous_targets = {}
            binary_targets = {}
            for feature, _ in decoder.discrete_features.items():
                discrete_targets[feature] = torch.ones(noisy_real.shape[0])

            for feature in decoder.continuous_features:
                continuous_targets[feature] = torch.ones(noisy_real.shape[0])

            for feature in decoder.binary_features:
                binary_targets[feature] = torch.ones(noisy_real.shape[0])

            optimizer_G.zero_grad()
            encoded = encoder_generator(noisy_real)
            dec_input = torch.cat([encoded, y], dim=1)
            discrete_outputs, continuous_outputs, binary_outputs = decoder.disc_cont(dec_input)

            g_loss = (0.1 * binary_cross_entropy(discriminator(encoded), valid) +
                      0.9 * decoder.compute_loss((discrete_outputs, continuous_outputs, binary_outputs),
                                                   (discrete_targets, continuous_targets, binary_targets)))

            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            z = AAE_archi_opt.custom_dist((real.shape[0], AAE_archi_opt.z_dim)).cuda() if cuda else AAE_archi_opt.custom_dist((real.shape[0], AAE_archi_opt.z_dim))
            real_loss = binary_cross_entropy(discriminator(z), (valid * (1 - 0.1) + 0.5 * 0.1))
            fake_loss = binary_cross_entropy(discriminator(encoded.detach()), (fake * (1 - 0.1) + 0.5 * 0.1))
            # l1_pen_d = l1_reg(discriminator, 0.0001)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

            # scheduler_G.step(g_loss)
        print(epoch, d_loss.item(), g_loss.item())
            # scheduler_D.step(d_loss)


    return epoch, d_loss.item(), g_loss.item()


"""-------------------------------------------------model validation-------------------------------------------------"""

def evaluate_model(val_loader):
    encoder_generator.eval()
    decoder.eval()
    discriminator.eval()

    total_g_loss = 0.0
    total_d_loss = 0.0

    with torch.no_grad():
        for X, y in val_loader:
            valid = torch.ones((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.ones((X.shape[0], 1),
                                                                                                    requires_grad=False)
            fake = torch.zeros((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.zeros((X.shape[0], 1),
                                                                                                     requires_grad=False)

            real = X.type(Tensor).cuda() if cuda else X.type(Tensor)
            y = y.type(Tensor).cuda() if cuda else y.type(Tensor)

            discrete_targets = {}
            continuous_targets = {}
            binary_targets = {}

            for feature, _ in decoder.discrete_features.items():
                discrete_targets[feature] = torch.ones(real.shape[0])

            for feature in decoder.continuous_features:
                continuous_targets[feature] = torch.ones(real.shape[0])

            for feature in decoder.binary_features:
                binary_targets[feature] = torch.ones(real.shape[0])

            encoded = encoder_generator(real)
            dec_input = torch.cat([encoded, y], dim=1)
            discrete_outputs, continuous_outputs, binary_outputs = decoder.disc_cont(dec_input)

            g_loss = (0.1 * binary_cross_entropy(discriminator(encoded),
                                                 torch.ones((X.shape[0], 1),
                                                            requires_grad=False).cuda() if cuda else torch.ones(
                                                     (X.shape[0], 1), requires_grad=False)) +
                      0.9 * decoder.compute_loss((discrete_outputs, continuous_outputs, binary_outputs),
                                                 (discrete_targets, continuous_targets, binary_targets)))

            total_g_loss += g_loss.item()

            z = AAE_archi_opt.custom_dist(
                (real.shape[0], AAE_archi_opt.z_dim)).cuda() if cuda else AAE_archi_opt.custom_dist(
                (real.shape[0], AAE_archi_opt.z_dim))
            real_loss = binary_cross_entropy(discriminator(z), valid)
            fake_loss = binary_cross_entropy(discriminator(encoded.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

        avg_g_loss = total_g_loss / len(val_loader)
        avg_d_loss = total_d_loss / len(val_loader)

    metrics = {
        'avg_g_loss': avg_g_loss,
        'avg_d_loss': avg_d_loss
    }

    return metrics




