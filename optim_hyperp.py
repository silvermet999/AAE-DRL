import json

import torch
import torch.nn as nn
import optuna
import itertools

import AAE_archi_opt

from optuna.trial import TrialState
import numpy as np
from AAE import AAE_archi_opt
cuda = True if torch.cuda.is_available() else False


class Encoder(nn.Module):
    def __init__(self, trial, in_features_e, z_dim):
        super(Encoder, self).__init__()
        layers_e = []
        n_layers_e = trial.suggest_int("n_layers_e", 1, 6)
        self.z_dim = z_dim

        for i in range(n_layers_e):
            out_features = trial.suggest_int(f"n_units_e_l{i}", z_dim+1, 26)
            layers_e.append(nn.Linear(in_features_e, out_features))
            layers_e.append(nn.LeakyReLU())
            layers_e.append(nn.BatchNorm1d(out_features))
            p_e = trial.suggest_float(f"dropout_l{i}", 0, 0.3)
            layers_e.append(nn.Dropout(p_e))
            in_features_e = out_features

        self.encoder_layers = nn.Sequential(*layers_e)
        self.mu_layer = nn.Linear(in_features_e, z_dim)
        self.logvar_layer = nn.Linear(in_features_e, z_dim)

    def forward(self, x):
        z_dim = self.z_dim
        x = self.encoder_layers(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        z = reparameterization(mu, logvar, z_dim)
        return z

def reparameterization(mu, logvar, z_dim):
    std = torch.exp(logvar / 2)
    device = mu.device
    sampled_z = torch.normal(0, 1, (mu.size(0), z_dim)).to(device)
    z = mu + std * sampled_z
    return z


class Decoder(nn.Module):
    def __init__(self, trial, in_features_de, z_dim):
        super(Decoder, self).__init__()
        layers_de = []
        n_layers_de = trial.suggest_int("n_layers_de", 1, 6)

        for i in range(n_layers_de):
            out_features = trial.suggest_int(f"n_units_de_l{i}", z_dim+1, 26)
            layers_de.append(nn.Linear(in_features_de, out_features))
            layers_de.append(nn.LeakyReLU())
            layers_de.append(nn.BatchNorm1d(out_features))
            p = trial.suggest_float(f"dropout_l{i}", 0, 0.5)
            layers_de.append(nn.Dropout(p))
            in_features_de = out_features
        layers_de.append(nn.Linear(in_features_de, 26))
        layers_de.append(nn.Tanh())
        self.decoder = nn.Sequential(*layers_de)

    def forward(self, x):
        return self.decoder(x)

class Discriminator(nn.Module):
    def __init__(self, trial, in_features_d, z_dim):
        super(Discriminator, self).__init__()
        layers_d = []
        n_layers_d = trial.suggest_int("n_layers_de", 1, 6)

        for i in range(n_layers_d):
            out_features = trial.suggest_int(f"n_units_de_l{i}", 1, z_dim-1)
            layers_d.append(nn.Linear(in_features_d, out_features))
            layers_d.append(nn.LeakyReLU())
            layers_d.append(nn.BatchNorm1d(out_features))
            p = trial.suggest_float(f"dropout_l{i}", 0, 0.5)
            layers_d.append(nn.Dropout(p))
            in_features_d = out_features
        layers_d.append(nn.Linear(in_features_d, 1))
        layers_d.append(nn.Sigmoid())
        self.discriminator = nn.Sequential(*layers_d)

    def forward(self, x):
        return self.discriminator(x)


def objective(trial):
    z_dim = trial.suggest_int("z_dim", 13, 24)
    in_features_de = z_dim + 12
    in_features_d = z_dim

    enc = Encoder(trial, 26, z_dim).cuda() if cuda else Encoder(trial, 26, z_dim)

    dec = Decoder(trial, in_features_de, z_dim).cuda() if cuda else Decoder(trial, in_features_de, z_dim)
    disc = Discriminator(trial, in_features_d, z_dim).cuda() if cuda else Discriminator(trial, in_features_d, z_dim)
    # enc = AAE_archi_opt.encoder_generator
    # dec = AAE_archi_opt.decoder
    # disc = AAE_archi_opt.discriminator
    adversarial_loss = nn.BCELoss().cuda() if cuda else nn.BCELoss()
    recon_loss = nn.L1Loss().cuda() if cuda else nn.L1Loss()
    lr = trial.suggest_float('lr', 1e-5, 0.01, log=True)
    beta1 = trial.suggest_float('beta1', 0.5, 0.9)
    beta2 = trial.suggest_float('beta2', 0.9, 0.999)
    lrd = trial.suggest_float('lrd', 1e-5, 0.01, log=True)
    beta1d = trial.suggest_float('beta1d', 0.5, 0.9)
    beta2d = trial.suggest_float('beta2d', 0.9, 0.999)
    optimizer_G = torch.optim.Adam(itertools.chain(enc.parameters(), dec.parameters()), lr=lr, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(disc.parameters(), lr=lrd, betas=(beta1d, beta2d))
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for epoch in range(100):
        for i, (X, y) in enumerate(AAE_archi_opt.dataloader):
            valid = torch.ones((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.ones((X.shape[0], 1),
                                                                                                    requires_grad=False)
            fake = torch.zeros((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.zeros((X.shape[0], 1),
                                                                                                     requires_grad=False)
            real = X.type(Tensor).cuda() if cuda else X.type(Tensor)
            y = y.type(Tensor).cuda() if cuda else y.type(Tensor)
            real = (real - real.mean()) / real.std()
            noisy_real = real + torch.randn_like(real) * 0.4

            optimizer_G.zero_grad()
            encoded = enc(noisy_real)
            dec_input = torch.cat([encoded, y], dim=1)
            decoded = dec(dec_input)
            g_loss = 0.001 * adversarial_loss(disc(encoded), valid) + 0.999 * recon_loss(decoded,
                                                                                                  noisy_real)

            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            z = torch.normal(0, 1, (real.shape[0], z_dim)).cuda() if cuda else torch.normal(0, 1, (
            real.shape[0], z_dim))
            real_loss = adversarial_loss(disc(z), valid)
            fake_loss = adversarial_loss(disc(encoded.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

    return np.mean(g_loss.item()), np.mean(d_loss.item())



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
    
    
