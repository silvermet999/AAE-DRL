import json

import torch
import torch.nn as nn
from torch.distributions import Normal
import optuna
import itertools

import main
import AAE_archi

from optuna.trial import TrialState
import torch.optim as optim
from sklearn.model_selection import KFold
import numpy as np

cuda = True if torch.cuda.is_available() else False

# def reparameterization(mu, logvar, z_dim):
#     std = torch.exp(logvar / 2)
#     device = mu.device
#     sampled_z = torch.normal(0, 1, (mu.size(0), z_dim)).to(device)
#     z = sampled_z * std + mu
#     return z

# class Encoder(nn.Module):
#     def __init__(self, trial, n_layers_e, in_features_e):
#         super(Encoder, self).__init__()
#         layers_e = []
#
#         for i in range(n_layers_e):
#             out_features = trial.suggest_int(f"n_units_e_l{i}", 32, 512)
#             layers_e.append(nn.Linear(in_features_e, out_features))
#             layers_e.append(nn.LeakyReLU())
#             layers_e.append(nn.BatchNorm1d(out_features))
#             p = trial.suggest_float(f"dropout_l{i}", 0, 0.5)
#             layers_e.append(nn.Dropout(p))
#             in_features_e = out_features
#
#         self.encoder = nn.Sequential(*layers_e)
#         self.mu_layer = nn.Linear(in_features_e, 32)
#         self.logvar_layer = nn.Linear(in_features_e, 32)
#
#     def forward(self, x):
#         x = self.encoder(x)
#         mu = self.mu_layer(x)
#         logvar = self.logvar_layer(x)
#         z = reparameterization(mu, logvar, 32)
#         return z

    #     for i in range(n_layers_e):
    #         out_features = trial.suggest_int(f"n_units_e_l{i}", 32, 512)
    #         layers_e.append(nn.Linear(in_features_e, out_features))
    #         layers_e.append(nn.LeakyReLU())
    #         layers_e.append(nn.BatchNorm1d(out_features))
    #         p = trial.suggest_float(f"dropout_l{i}", 0, 0.5)
    #         layers_e.append(nn.Dropout(p))
    #         in_features_e = out_features
    #
    #     self.encoder = nn.Sequential(*layers_e)
    #     self.mu_layer = nn.Linear(in_features_e, 32)
    #     self.logvar_layer = nn.Linear(in_features_e, 32)
    #
    # def forward(self, x):
    #     x = self.encoder(x)
    #     mu = self.mu_layer(x)
    #     logvar = self.logvar_layer(x)
    #     z = reparameterization(mu, logvar, 32)
    #     return z


# class Decoder(nn.Module):
#     def __init__(self, trial, n_layers_de, in_features_de):
#         super(Decoder, self).__init__()
#         layers_de = []
#
#         for i in range(n_layers_de):
#             out_features = trial.suggest_int(f"n_units_de_l{i}", 32, 512)
#             layers_de.append(nn.Linear(in_features_de, out_features))
#             layers_de.append(nn.LeakyReLU())
#             layers_de.append(nn.BatchNorm1d(out_features))
    #         p = trial.suggest_float("dropout_l{}".format(i), 0, 0.5)
    #         layers_de.append(nn.Dropout(p))
    #         in_features_de = out_features
    #     layers_de.append(nn.Linear(in_features_de, 105))
    #     layers_de.append(nn.Tanh())
    #     self.decoder = nn.Sequential(*layers_de)
    #
    # def forward(self, x):
    #     return self.decoder(x)

#
# class Discriminator(nn.Module):
#     def __init__(self, trial, n_layers_di, in_features_di):
#         super(Discriminator, self).__init__()
#         layers_di = []
#
#         for i in range(n_layers_di):
#             out_features = trial.suggest_int(f"n_units_di_l{i}", 32, 512)
#             layers_di.append(nn.Linear(in_features_di, out_features))
#             layers_di.append(nn.LeakyReLU())
#             layers_di.append(nn.BatchNorm1d(out_features))
#             p = trial.suggest_float("dropout_l{}".format(i), 0, 0.5)
#             layers_di.append(nn.Dropout(p))
#             in_features_di = out_features
#         layers_di.append(nn.Linear(in_features_di, 1))
#         layers_di.append(nn.Sigmoid())
#         self.discriminator = nn.Sequential(*layers_di)
#
#     def forward(self, x):
#         return self.discriminator(x)


def define_model(trial):
    # n_layers_e = trial.suggest_int("n_layers_e", 20, 40)
    # n_layers_de = trial.suggest_int("n_layers_de", 20, 40)
    # n_layers_di = trial.suggest_int("n_layers_di", 10, 40)
    # in_features_e = 105
    # in_features_de = 32
    # in_features_di = 32
    #
    # encoder = Encoder(trial, n_layers_e, in_features_e).cuda() if cuda else Encoder(trial, n_layers_e, in_features_e)
    # decoder = Decoder(trial, n_layers_de, in_features_de).cuda() if cuda else Decoder(trial, n_layers_de, in_features_de)
    encoder = AAE_archi.EncoderGenerator().cuda()
    decoder = AAE_archi.Decoder().cuda()
    discriminator = AAE_archi.Discriminator().cuda()
    # discriminator = Discriminator(trial, n_layers_di, in_features_di).cuda() if cuda else Discriminator(trial, n_layers_di, in_features_di)

    return encoder, decoder, discriminator





def objective(trial):
    enc, dec, disc = define_model(trial)
    adversarial_loss = nn.BCELoss()
    recon_loss = nn.L1Loss().cuda()
    #
    lr = trial.suggest_float('lr', 1e-5, 0.1, log=True)
    beta1 = trial.suggest_float('beta1', 0.5, 0.9)
    beta2 = trial.suggest_float('beta2', 0.9, 0.999)
    lrd = trial.suggest_float('lrd', 1e-5, 0.1, log=True)
    beta1d = trial.suggest_float('beta1d', 0.5, 0.9)
    beta2d = trial.suggest_float('beta2d', 0.9, 0.999)
    optimizer_G = torch.optim.Adam(itertools.chain(enc.parameters(), dec.parameters()), lr=lr, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(disc.parameters(), lr=lrd, betas=(beta1d, beta2d))

    train = main.X_train_rs


    for epoch in range(100):
        n_batch = len(train) // 32
        for i in range(n_batch):
            str_idx = i * 32
            end_idx = str_idx + 32
            batch_data = train[str_idx:end_idx]

            train_data_tensor = torch.tensor(batch_data, dtype=torch.float).cuda() if cuda else torch.tensor(batch_data, dtype=torch.float)
            real = (train_data_tensor - train_data_tensor.mean()) / train_data_tensor.std()
            valid = torch.ones((train_data_tensor.shape[0], 1)).cuda() if cuda else torch.ones((train_data_tensor.shape[0], 1))
            fake = torch.zeros((train_data_tensor.shape[0], 1)).cuda() if cuda else torch.zeros((train_data_tensor.shape[0], 1))

            optimizer_G.zero_grad()
            encoded = enc(real)
            labels_tensor = AAE_archi.encoded_tensor[str_idx:end_idx]
            dec_input = torch.cat([encoded, labels_tensor], dim=1)
            decoded = dec(dec_input)
            g_loss = 0.001 * adversarial_loss(disc(encoded), valid)+0.999 * recon_loss(decoded, real)

            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            z = torch.normal(0, 1, (batch_data.shape[0], 32)).cuda() if cuda else torch.normal(0, 1, (batch_data.shape[0], 32))

            # real and fake loss should be close
            # discriminator(z) should be close to 0
            real_loss = adversarial_loss(disc(z), valid)
            fake_loss = adversarial_loss(disc(encoded.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()


    if epoch == 99:
        trial.report(g_loss.item(), d_loss.item())
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()


    return g_loss, d_loss



study = optuna.create_study(directions=["minimize", "minimize"])
study.optimize(objective, n_trials=50)
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of complete trials: ", len(complete_trials))

print("Pareto front:")
for trial in study.best_trials:
    print("  Trial number: ", trial.number)
    print("  Values: ", trial.values)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print()

with open('layers+params.json', 'w') as f:
    json.dump(study.best_trials, f, indent=4)
