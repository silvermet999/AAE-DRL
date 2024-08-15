import json
import os
import torch
import torch.nn as nn
import optuna
import itertools

import main

from optuna.trial import TrialState
import torch.optim as optim
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

cuda = True if torch.cuda.is_available() else False

def reparameterization(mu, logvar, z_dim):
    std = torch.exp(logvar / 2)
    device = mu.device
    sampled_z = torch.normal(0, 1, (mu.size(0), z_dim)).to(device)
    z = sampled_z * std + mu
    return z

class Encoder(nn.Module):
    def __init__(self, trial, n_layers_e, in_features_e):
        super(Encoder, self).__init__()

        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        device = torch.device(f'cuda:{rank}')

        layers_e = []

        for i in range(n_layers_e):
            out_features = trial.suggest_int(f"n_units_e_l{i}", 32, 512)
            layers_e.append(nn.Linear(in_features_e, out_features))
            layers_e.append(nn.LeakyReLU())
            layers_e.append(nn.BatchNorm1d(out_features))
            p = trial.suggest_float(f"dropout_l{i}", 0, 0.5)
            layers_e.append(nn.Dropout(p))
            in_features_e = out_features

        self.encoder = nn.Sequential(*layers_e).to(device)
        self.mu_layer = nn.Linear(in_features_e, 32)
        self.logvar_layer = nn.Linear(in_features_e, 32)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        z = reparameterization(mu, logvar, 32)
        return z


class Decoder(nn.Module):
    def __init__(self, trial, n_layers_de, in_features_de):
        super(Decoder, self).__init__()
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        device = torch.device(f'cuda:{rank}')

        layers_de = []

        for i in range(n_layers_de):
            out_features = trial.suggest_int(f"n_units_de_l{i}", 32, 512)
            layers_de.append(nn.Linear(in_features_de, out_features))
            layers_de.append(nn.LeakyReLU())
            layers_de.append(nn.BatchNorm1d(out_features))
            p = trial.suggest_float("dropout_l{}".format(i), 0, 0.5)
            layers_de.append(nn.Dropout(p))
            in_features_de = out_features
        layers_de.append(nn.Linear(in_features_de, 119))
        layers_de.append(nn.Tanh())
        self.decoder = nn.Sequential(*layers_de).to(device)

    def forward(self, x):
        return self.decoder(x)


class Discriminator(nn.Module):
    def __init__(self, trial, n_layers_di, in_features_di):
        super(Discriminator, self).__init__()
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        device = torch.device(f'cuda:{rank}')

        layers_di = []

        for i in range(n_layers_di):
            out_features = trial.suggest_int(f"n_units_di_l{i}", 32, 512)
            layers_di.append(nn.Linear(in_features_di, out_features))
            layers_di.append(nn.LeakyReLU())
            layers_di.append(nn.BatchNorm1d(out_features))
            p = trial.suggest_float("dropout_l{}".format(i), 0, 0.5)
            layers_di.append(nn.Dropout(p))
            in_features_di = out_features
        layers_di.append(nn.Linear(in_features_di, 1))
        layers_di.append(nn.Sigmoid())
        self.discriminator = nn.Sequential(*layers_di).to(device)

    def forward(self, x):
        return self.discriminator(x)


def define_model(rank, trial):
    n_layers_e = trial.suggest_int("n_layers_e", 20, 50)
    n_layers_de = trial.suggest_int("n_layers_de", 20, 50)
    n_layers_di = trial.suggest_int("n_layers_di", 10, 30)
    in_features_e = 119
    in_features_de = 32
    in_features_di = 32

    encoder = Encoder(trial, n_layers_e, in_features_e)
    encoder = DDP(encoder, device_ids=[rank])
    decoder = Decoder(trial, n_layers_de, in_features_de)
    decoder = DDP(decoder, device_ids=[rank])
    discriminator = Discriminator(trial, n_layers_di, in_features_di)
    discriminator = DDP(discriminator, device_ids=[rank])

    return encoder, decoder, discriminator





def objective(trial):
    enc, dec, disc = define_model(trial)
    adversarial_loss = nn.BCELoss()
    recon_loss = nn.L1Loss()

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = trial.suggest_float('lr', 1e-5, 0.1, log=True)
    beta1 = trial.suggest_float('beta1', 0.5, 0.9)
    beta2 = trial.suggest_float('beta2', 0.9, 0.999)
    lrd = trial.suggest_float('lrd', 1e-5, 0.1, log=True)
    beta1d = trial.suggest_float('beta1d', 0.5, 0.9)
    beta2d = trial.suggest_float('beta2d', 0.9, 0.999)
    optimizer_G = getattr(optim, optimizer_name)(itertools.chain(enc.parameters(), dec.parameters()), lr=lr,
                                                 betas=(beta1, beta2))
    optimizer_D = getattr(optim, optimizer_name)(disc.parameters(), lr=lrd, betas=(beta1d, beta2d))

    train = main.df_n
    test = main.df_t
    recon = []
    adv = []


    for epoch in range(200):
        n_batch = len(train) // 32
        for i in range(n_batch):
            str_idx = i * 32
            end_idx = str_idx + 32
            batch_data = train.iloc[str_idx:end_idx].values

            train_data_tensor = torch.tensor(batch_data, dtype=torch.float).cuda() if cuda else torch.tensor(batch_data, dtype=torch.float)
            real = (train_data_tensor - train_data_tensor.mean()) / train_data_tensor.std()
            valid = torch.ones((train_data_tensor.shape[0], 1)).cuda() if cuda else torch.ones((train_data_tensor.shape[0], 1))
            fake = torch.zeros((train_data_tensor.shape[0], 1)).cuda() if cuda else torch.zeros((train_data_tensor.shape[0], 1))

            optimizer_G.zero_grad()
            encoded = enc(real)
            decoded = dec(encoded)
            g_loss = 0.001 * adversarial_loss(disc(encoded), valid) + 0.999 * recon_loss(decoded, real)

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

        enc.eval()
        dec.eval()
        disc.eval()
        n_batch_t = len(test) // 1
        for i in range(n_batch_t):
            str_idx = i * 1
            end_idx = str_idx + 1
            with torch.no_grad():
                val_tensor = torch.tensor(test.iloc[str_idx:end_idx].values, dtype=torch.float).cuda() if cuda else torch.tensor(test.iloc[str_idx:end_idx].values, dtype=torch.float)
                val_real = (val_tensor - val_tensor.mean()) / val_tensor.std()
                val_encoded = enc(val_real)
                val_decoded = dec(val_encoded)
            recon_loss_val = recon_loss(val_decoded, val_real)
            valid_val = torch.ones((val_real.shape[0], 1)).cuda() if cuda else torch.ones((val_real.shape[0], 1))
            adv_loss_val = adversarial_loss(disc(val_encoded), valid_val)
        recon.append(np.mean(recon_loss_val.item()))
        adv.append(np.mean(adv_loss_val.item()))
    recon_mean = np.mean(recon)
    adv_mean = np.mean(adv)

    return recon_mean, adv_mean

if __name__ == "__main__":
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=10)

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

# [I 2024-08-10 15:40:57,666] Trial 0 finished with values: [0.09702626450194253, 0.039081158461545716] and parameters: {'n_layers_e': 44, 'n_layers_de': 37, 'n_layers_di': 11, 'n_units_e_l0': 369, 'dropout_l0': 0.2912014074950425, 'n_units_e_l1': 264, 'dropout_l1': 0.35957829905370775, 'n_units_e_l2': 143, 'dropout_l2': 0.16634929998161474, 'n_units_e_l3': 418, 'dropout_l3': 0.14488264346896107, 'n_units_e_l4': 483, 'dropout_l4': 0.34961088800051693, 'n_units_e_l5': 340, 'dropout_l5': 0.2882157846404448, 'n_units_e_l6': 374, 'dropout_l6': 0.02374574253753159, 'n_units_e_l7': 185, 'dropout_l7': 0.2984299208956127, 'n_units_e_l8': 152, 'dropout_l8': 0.07317829255139408, 'n_units_e_l9': 225, 'dropout_l9': 0.4633390011826975, 'n_units_e_l10': 248, 'dropout_l10': 0.18925529370010746, 'n_units_e_l11': 406, 'dropout_l11': 0.04124974957292793, 'n_units_e_l12': 501, 'dropout_l12': 0.3050765260201068, 'n_units_e_l13': 252, 'dropout_l13': 0.4529985369856203, 'n_units_e_l14': 392, 'dropout_l14': 0.09256666951563924, 'n_units_e_l15': 313, 'dropout_l15': 0.0034110266440671166, 'n_units_e_l16': 166, 'dropout_l16': 0.2433992663000255, 'n_units_e_l17': 54, 'dropout_l17': 0.3637684362506816, 'n_units_e_l18': 142, 'dropout_l18': 0.11901943868375703, 'n_units_e_l19': 224, 'dropout_l19': 0.12485899981978027, 'n_units_e_l20': 321, 'dropout_l20': 0.40862458111926414, 'n_units_e_l21': 493, 'dropout_l21': 0.25096111389024267, 'n_units_e_l22': 303, 'dropout_l22': 0.17344412654847136, 'n_units_e_l23': 153, 'dropout_l23': 0.19765134545072033, 'n_units_e_l24': 138, 'dropout_l24': 0.0293149257260214, 'n_units_e_l25': 252, 'dropout_l25': 0.43794389274878254, 'n_units_e_l26': 110, 'dropout_l26': 0.4365971670818343, 'n_units_e_l27': 376, 'dropout_l27': 0.4932216553741478, 'n_units_e_l28': 245, 'dropout_l28': 0.17462278881475202, 'n_units_e_l29': 349, 'dropout_l29': 0.04790680468013431, 'n_units_e_l30': 478, 'dropout_l30': 0.189794748643973, 'n_units_e_l31': 424, 'dropout_l31': 0.28708515210653035, 'n_units_e_l32': 63, 'dropout_l32': 0.12792539876826858, 'n_units_e_l33': 204, 'dropout_l33': 0.04126183161713565, 'n_units_e_l34': 438, 'dropout_l34': 0.46693733936443116, 'n_units_e_l35': 390, 'dropout_l35': 0.16244944823482776, 'n_units_e_l36': 456, 'dropout_l36': 0.3438666329016294, 'n_units_e_l37': 294, 'dropout_l37': 0.3571553789563026, 'n_units_e_l38': 348, 'dropout_l38': 0.30372651918722593, 'n_units_e_l39': 239, 'dropout_l39': 0.3367211118052325, 'n_units_e_l40': 479, 'dropout_l40': 0.44870344468741424, 'n_units_e_l41': 260, 'dropout_l41': 0.00375152328986611, 'n_units_e_l42': 457, 'dropout_l42': 0.34571830645315565, 'n_units_e_l43': 310, 'dropout_l43': 0.3001775340913507, 'n_units_de_l0': 225, 'n_units_de_l1': 281, 'n_units_de_l2': 303, 'n_units_de_l3': 444, 'n_units_de_l4': 59, 'n_units_de_l5': 124, 'n_units_de_l6': 245, 'n_units_de_l7': 293, 'n_units_de_l8': 442, 'n_units_de_l9': 444, 'n_units_de_l10': 494, 'n_units_de_l11': 438, 'n_units_de_l12': 414, 'n_units_de_l13': 83, 'n_units_de_l14': 255, 'n_units_de_l15': 420, 'n_units_de_l16': 315, 'n_units_de_l17': 473, 'n_units_de_l18': 52, 'n_units_de_l19': 161, 'n_units_de_l20': 501, 'n_units_de_l21': 88, 'n_units_de_l22': 439, 'n_units_de_l23': 441, 'n_units_de_l24': 257, 'n_units_de_l25': 313, 'n_units_de_l26': 472, 'n_units_de_l27': 326, 'n_units_de_l28': 382, 'n_units_de_l29': 187, 'n_units_de_l30': 188, 'n_units_de_l31': 248, 'n_units_de_l32': 98, 'n_units_de_l33': 158, 'n_units_de_l34': 35, 'n_units_de_l35': 236, 'n_units_de_l36': 233, 'n_units_di_l0': 493, 'n_units_di_l1': 445, 'n_units_di_l2': 64, 'n_units_di_l3': 290, 'n_units_di_l4': 359, 'n_units_di_l5': 421, 'n_units_di_l6': 403, 'n_units_di_l7': 118, 'n_units_di_l8': 162, 'n_units_di_l9': 134, 'n_units_di_l10': 466, 'optimizer': 'Adam', 'lr': 1.802133395290409e-05, 'beta1': 0.8077587688567314, 'beta2': 0.980815082090427, 'lrd': 0.001290574059129973, 'beta1d': 0.8339287336360124, 'beta2d': 0.9128002537599588, 'n_ep': 54, 'batch': 99}.