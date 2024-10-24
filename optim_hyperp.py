import json

import torch
import torch.nn as nn
import optuna
import itertools

from data import main
import AAE_archi

from optuna.trial import TrialState
import numpy as np

cuda = True if torch.cuda.is_available() else False


class Encoder(nn.Module):
    def __init__(self, trial, in_features_e):
        super(Encoder, self).__init__()
        layers_e = []
        n_layers_e = trial.suggest_int("n_layers_e", 5, 20)
        z_dim = 40

        for i in range(n_layers_e):
            out_features = trial.suggest_int(f"n_units_e_l{i}", 5, 50)
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
        x = self.encoder_layers(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        z = reparameterization(mu, logvar, 40)
        return z

def reparameterization(mu, logvar, z_dim):
    std = torch.exp(logvar / 2)
    device = mu.device
    sampled_z = torch.normal(0, 1, (mu.size(0), z_dim)).to(device)
    z = mu + std * sampled_z
    return z


class Decoder(nn.Module):
    def __init__(self, trial, in_features_de):
        super(Decoder, self).__init__()
        layers_de = []
        n_layers_de = trial.suggest_int("n_layers_de", 5, 20)

        for i in range(n_layers_de):
            out_features = trial.suggest_int(f"n_units_de_l{i}", 5, 50)
            layers_de.append(nn.Linear(in_features_de, out_features))
            layers_de.append(nn.LeakyReLU())
            layers_de.append(nn.BatchNorm1d(out_features))
            p = trial.suggest_float(f"dropout_l{i}", 0, 0.5)
            layers_de.append(nn.Dropout(p))
            in_features_de = out_features

        layers_de.append(nn.Linear(in_features_de, 46))
        layers_de.append(nn.Tanh())

        self.decoder = nn.Sequential(*layers_de)

    def forward(self, x):
        return self.decoder(x)

class Discriminator(nn.Module):
    def __init__(self, trial, in_features_d):
        super(Discriminator, self).__init__()
        layers_d = []
        n_layers_d = trial.suggest_int("n_layers_de", 5, 20)

        for i in range(n_layers_d):
            out_features = trial.suggest_int(f"n_units_de_l{i}", 5, 50)
            layers_d.append(nn.Linear(in_features_d, out_features))
            layers_d.append(nn.LeakyReLU())
            layers_d.append(nn.BatchNorm1d(out_features))
            p = trial.suggest_float(f"dropout_l{i}", 0, 0.5)
            layers_d.append(nn.Dropout(p))
            in_features_d = out_features
        layers_d.append(nn.Linear(in_features_d, 1))
        layers_d.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*layers_d)

    def forward(self, x):
        return self.decoder(x)



def objective(trial):
    in_features_e = 46
    in_features_de = 40 + 12
    in_features_d = 40

    enc = Encoder(trial, in_features_e).cuda() if cuda else Encoder(trial, in_features_e)
    dec = Decoder(trial, in_features_de).cuda() if cuda else Decoder(trial, in_features_de)
    disc = Discriminator(trial, in_features_d).cuda() if cuda else Discriminator(trial, in_features_d)
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

    recon_losses = []
    adversarial_losses = []

    epochs = trial.suggest_int("epochs", 100, 300)
    batches = trial.suggest_int("batches", 32, 128)
    encoded_tensor = torch.tensor(main.y_train.values, dtype=torch.float32).cuda() if cuda else torch.tensor(
        main.y_train.values, dtype=torch.float32)
    df_train = main.X_train_rs
    for epoch in range(epochs):
        n_batch = len(df_train) // batches
        for i in range(n_batch):
            str_idx = i * batches
            end_idx = str_idx + batches
            train_data_tensor = torch.tensor(df_train[str_idx:end_idx], dtype=torch.float).cuda() if cuda \
                else torch.tensor(df_train[str_idx:end_idx], dtype=torch.float)
            real = (train_data_tensor - train_data_tensor.mean()) / train_data_tensor.std()
            valid = torch.ones((train_data_tensor.shape[0], 1)).cuda() if cuda else torch.ones(
                (train_data_tensor.shape[0], 1))
            fake = torch.zeros((train_data_tensor.shape[0], 1)).cuda() if cuda else torch.zeros(
                (train_data_tensor.shape[0], 1))

            optimizer_G.zero_grad()
            encoded = enc(real)
            labels_tensor = encoded_tensor[str_idx:end_idx]
            dec_input = torch.cat([encoded, labels_tensor], dim=1)
            decoded = dec(dec_input)
            g_loss = 0.001 * adversarial_loss(disc(encoded), valid)+0.999 * recon_loss(decoded, real)

            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            z = torch.normal(0, 1, (train_data_tensor.shape[0], in_features_d)).cuda() if cuda else torch.normal(0, 1, (train_data_tensor.shape[0], in_features_d))
            real_loss = adversarial_loss(disc(z), valid)
            fake_loss = adversarial_loss(disc(encoded.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

    enc.eval()
    dec.eval()
    disc.eval()
    n_batch = len(df_train)
    for i in range(n_batch):
        str_idx = i * 1
        end_idx = str_idx + 1
        with torch.no_grad():
            val_tensor = torch.tensor(df_train[str_idx:end_idx], dtype=torch.float).cuda() if cuda else torch.tensor(
                df_train[str_idx:end_idx], dtype=torch.float)
            val_real = (val_tensor - val_tensor.mean()) / val_tensor.std()
            val_encoded = enc(val_real)
            labels_tensor = AAE_archi.encoded_tensor[str_idx:end_idx]
            dec_input = torch.cat([val_encoded, labels_tensor], dim=1)
            val_decoded = dec(dec_input)
        recon_loss_val = recon_loss(val_decoded, val_real)
        valid_val = torch.ones((val_real.shape[0], 1)).cuda() if cuda else torch.ones((val_real.shape[0], 1))
        adv_loss_val = adversarial_loss(disc(val_encoded), valid_val)
        recon_losses.append(np.mean(recon_loss_val.item()))
        adversarial_losses.append(np.mean(adv_loss_val.item()))
    avg_recon_loss = np.mean(recon_losses)
    avg_adversarial_loss = np.mean(adversarial_losses)


    return avg_recon_loss, avg_adversarial_loss



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
    
    
