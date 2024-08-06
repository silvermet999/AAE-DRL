import torch
import torch.nn as nn
from torch.distributions import Normal
import optuna
import itertools

import main

from optuna.trial import TrialState
import torch.optim as optim
from sklearn.model_selection import KFold
import numpy as np

def reparameterization(mu, logvar, z_dim):
    std = torch.exp(logvar / 2)
    device = mu.device
    sampled_z = torch.normal(0, 1, (mu.size(0), z_dim)).to(device)
    z = sampled_z * std + mu
    return z

class Encoder(nn.Module):
    def __init__(self, trial, n_layers_e, in_features_e):
        super(Encoder, self).__init__()
        layers_e = []

        for i in range(n_layers_e):
            out_features = trial.suggest_int(f"n_units_e_l{i}", 32, 512)
            layers_e.append(nn.Linear(in_features_e, out_features))
            layers_e.append(nn.LeakyReLU())
            layers_e.append(nn.BatchNorm1d(out_features))
            p = trial.suggest_float(f"dropout_l{i}", 0, 0.5)
            layers_e.append(nn.Dropout(p))
            in_features_e = out_features

        self.encoder = nn.Sequential(*layers_e)
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
        layers_de = []

        for i in range(n_layers_de):
            out_features = trial.suggest_int(f"n_units_de_l{i}", 32, 512)
            layers_de.append(nn.Linear(in_features_de, out_features))
            layers_de.append(nn.LeakyReLU())
            layers_de.append(nn.BatchNorm1d(out_features))
            p = trial.suggest_float("dropout_l{}".format(i), 0, 0.5)
            layers_de.append(nn.Dropout(p))
            in_features_de = out_features
        layers_de.append(nn.Linear(in_features_de, 117))
        layers_de.append(nn.Tanh())
        self.decoder = nn.Sequential(*layers_de)

    def forward(self, x):
        return self.decoder(x)


class Discriminator(nn.Module):
    def __init__(self, trial, n_layers_di, in_features_di):
        super(Discriminator, self).__init__()
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
        self.discriminator = nn.Sequential(*layers_di)

    def forward(self, x):
        return self.discriminator(x)


def define_model(trial):
    n_layers_e = trial.suggest_int("n_layers_e", 20, 50)
    n_layers_de = trial.suggest_int("n_layers_de", 20, 50)
    n_layers_di = trial.suggest_int("n_layers_di", 10, 40)
    in_features_e = 117
    in_features_de = 32
    in_features_di = 32

    encoder = Encoder(trial, n_layers_e, in_features_e)
    decoder = Decoder(trial, n_layers_de, in_features_de)
    discriminator = Discriminator(trial, n_layers_di, in_features_di)

    return encoder, decoder, discriminator





def objective(trial):
    enc, dec, disc = define_model(trial)
    adversarial_loss = nn.BCELoss()
    recon_loss = nn.L1Loss()

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    beta1 = trial.suggest_float('beta1', 0.5, 0.9)
    beta2 = trial.suggest_float('beta2', 0.9, 0.999)
    lrd = trial.suggest_float('lrd', 1e-5, 1e-1, log=True)
    beta1d = trial.suggest_float('beta1d', 0.5, 0.9)
    beta2d = trial.suggest_float('beta2d', 0.9, 0.999)
    optimizer_G = getattr(optim, optimizer_name)(itertools.chain(enc.parameters(), dec.parameters()), lr=lr,
                                                 betas=(beta1, beta2))
    optimizer_D = getattr(optim, optimizer_name)(disc.parameters(), lr=lrd, betas=(beta1d, beta2d))

    train = main.df_n
    # test = main.X_test_rs_cl
    # recon = []
    # adv = []
    n_ep = trial.suggest_int("n_ep", 50, 150)
    batch = trial.suggest_int("n_batch", 32, 128)


    for epoch in range(n_ep):
        n_batch = len(train) // batch
        for i in range(n_batch):
            str_idx = i * batch
            end_idx = str_idx + batch
            batch_data = train.iloc[str_idx:end_idx].values

            train_data_tensor = torch.tensor(batch_data, dtype=torch.float)
            real = (train_data_tensor - train_data_tensor.mean()) / train_data_tensor.std()
            valid = torch.ones((train_data_tensor.shape[0], 1))
            fake = torch.zeros((train_data_tensor.shape[0], 1))

            optimizer_G.zero_grad()
            encoded = enc(real)
            decoded = dec(encoded)
            g_loss = 0.001 * adversarial_loss(disc(encoded).sigmoid(), valid) + 0.999 * recon_loss(decoded, real)

            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            z = torch.normal(0, 1, (batch_data.shape[0], 32))

            # real and fake loss should be close
            # discriminator(z) should be close to 0
            real_loss = adversarial_loss(disc(z), valid)
            fake_loss = adversarial_loss(disc(encoded.detach().sigmoid()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

    return g_loss.item(), d_loss.item()



study = optuna.create_study(directions=["minimize", "minimize"])

print("started")
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


# [I 2024-08-05 17:53:13,049] Trial 0 finished with values: [0.08509857207536697, 0.6866985559463501] and parameters: {'n_layers_e': 40, 'n_layers_de': 49, 'n_layers_di': 21, 'n_units_e_l0': 81, 'dropout_l0': 0.10867360860505987, 'n_units_e_l1': 493, 'dropout_l1': 0.08939577742112104, 'n_units_e_l2': 427, 'dropout_l2': 0.29538653398998005, 'n_units_e_l3': 495, 'dropout_l3': 0.07350348642662907, 'n_units_e_l4': 56, 'dropout_l4': 0.1890892929708305, 'n_units_e_l5': 306, 'dropout_l5': 0.428471637373925, 'n_units_e_l6': 502, 'dropout_l6': 0.18351259292857713, 'n_units_e_l7': 290, 'dropout_l7': 0.06961080854845647, 'n_units_e_l8': 458, 'dropout_l8': 0.37943476919964647, 'n_units_e_l9': 415, 'dropout_l9': 0.0829691824603962, 'n_units_e_l10': 194, 'dropout_l10': 0.4966813821839796, 'n_units_e_l11': 460, 'dropout_l11': 0.22466212449268563, 'n_units_e_l12': 382, 'dropout_l12': 0.2248963910421939, 'n_units_e_l13': 192, 'dropout_l13': 0.25407714522905905, 'n_units_e_l14': 369, 'dropout_l14': 0.23111987729553496, 'n_units_e_l15': 408, 'dropout_l15': 0.2607985297237844, 'n_units_e_l16': 489, 'dropout_l16': 0.199656936142855, 'n_units_e_l17': 123, 'dropout_l17': 0.34988855521146606, 'n_units_e_l18': 55, 'dropout_l18': 0.4374361426532604, 'n_units_e_l19': 56, 'dropout_l19': 0.3579706431295556, 'n_units_e_l20': 304, 'dropout_l20': 0.21041676485133526, 'n_units_e_l21': 357, 'dropout_l21': 0.21131596989551665, 'n_units_e_l22': 298, 'dropout_l22': 0.045456286199973495, 'n_units_e_l23': 258, 'dropout_l23': 0.22144631289316036, 'n_units_e_l24': 425, 'dropout_l24': 0.45765814324753556, 'n_units_e_l25': 475, 'dropout_l25': 0.09593486737490392, 'n_units_e_l26': 485, 'dropout_l26': 0.2350393336122053, 'n_units_e_l27': 412, 'dropout_l27': 0.19220591415175936, 'n_units_e_l28': 230, 'dropout_l28': 0.28872727369455403, 'n_units_e_l29': 90, 'dropout_l29': 0.38638537382499344, 'n_units_e_l30': 301, 'dropout_l30': 0.23548303227726614, 'n_units_e_l31': 328, 'dropout_l31': 0.4272981265313643, 'n_units_e_l32': 423, 'dropout_l32': 0.23448998205192212, 'n_units_e_l33': 447, 'dropout_l33': 0.4903480550816319, 'n_units_e_l34': 388, 'dropout_l34': 0.23468202502318114, 'n_units_e_l35': 380, 'dropout_l35': 0.08000127188942019, 'n_units_e_l36': 155, 'dropout_l36': 0.16510041220074428, 'n_units_e_l37': 298, 'dropout_l37': 0.39773802798901225, 'n_units_e_l38': 174, 'dropout_l38': 0.34845333667371386, 'n_units_e_l39': 365, 'dropout_l39': 0.22106608101720981, 'n_units_de_l0': 304, 'n_units_de_l1': 215, 'n_units_de_l2': 376, 'n_units_de_l3': 340, 'n_units_de_l4': 435, 'n_units_de_l5': 463, 'n_units_de_l6': 387, 'n_units_de_l7': 426, 'n_units_de_l8': 397, 'n_units_de_l9': 327, 'n_units_de_l10': 251, 'n_units_de_l11': 48, 'n_units_de_l12': 152, 'n_units_de_l13': 507, 'n_units_de_l14': 283, 'n_units_de_l15': 433, 'n_units_de_l16': 112, 'n_units_de_l17': 125, 'n_units_de_l18': 110, 'n_units_de_l19': 72, 'n_units_de_l20': 50, 'n_units_de_l21': 67, 'n_units_de_l22': 289, 'n_units_de_l23': 469, 'n_units_de_l24': 374, 'n_units_de_l25': 498, 'n_units_de_l26': 411, 'n_units_de_l27': 243, 'n_units_de_l28': 124, 'n_units_de_l29': 481, 'n_units_de_l30': 169, 'n_units_de_l31': 374, 'n_units_de_l32': 476, 'n_units_de_l33': 263, 'n_units_de_l34': 278, 'n_units_de_l35': 456, 'n_units_de_l36': 193, 'n_units_de_l37': 322, 'n_units_de_l38': 324, 'n_units_de_l39': 249, 'n_units_de_l40': 420, 'dropout_l40': 0.4818833520610307, 'n_units_de_l41': 258, 'dropout_l41': 0.10924565086055182, 'n_units_de_l42': 487, 'dropout_l42': 0.11649344512924437, 'n_units_de_l43': 474, 'dropout_l43': 0.1643091599074476, 'n_units_de_l44': 429, 'dropout_l44': 0.47646904233147647, 'n_units_de_l45': 502, 'dropout_l45': 0.04335551667745008, 'n_units_de_l46': 106, 'dropout_l46': 0.45448049594566486, 'n_units_de_l47': 133, 'dropout_l47': 0.3358463706821003, 'n_units_de_l48': 352, 'dropout_l48': 0.23292333863339726, 'n_units_di_l0': 258, 'n_units_di_l1': 132, 'n_units_di_l2': 235, 'n_units_di_l3': 512, 'n_units_di_l4': 324, 'n_units_di_l5': 90, 'n_units_di_l6': 428, 'n_units_di_l7': 380, 'n_units_di_l8': 420, 'n_units_di_l9': 237, 'n_units_di_l10': 153, 'n_units_di_l11': 177, 'n_units_di_l12': 172, 'n_units_di_l13': 207, 'n_units_di_l14': 216, 'n_units_di_l15': 463, 'n_units_di_l16': 150, 'n_units_di_l17': 444, 'n_units_di_l18': 71, 'n_units_di_l19': 298, 'n_units_di_l20': 294, 'optimizer': 'Adam', 'lr': 0.0012529560964239877, 'beta1': 0.7444090539397276, 'beta2': 0.9210971752955273, 'lrd': 0.001156809765154699, 'beta1d': 0.6914175786200397, 'beta2d': 0.9011393110072864}.
# [I 2024-08-05 18:03:46,575] Trial 2 finished with values: [0.08275523036718369, 0.7051534652709961] and parameters: {'n_layers_e': 26, 'n_layers_de': 30, 'n_layers_di': 21, 'n_units_e_l0': 341, 'dropout_l0': 0.12489609769842458, 'n_units_e_l1': 100, 'dropout_l1': 0.11585228591187346, 'n_units_e_l2': 240, 'dropout_l2': 0.46611040777578555, 'n_units_e_l3': 475, 'dropout_l3': 0.12551279143949023, 'n_units_e_l4': 470, 'dropout_l4': 0.39514032428367507, 'n_units_e_l5': 105, 'dropout_l5': 0.28190402582514795, 'n_units_e_l6': 459, 'dropout_l6': 0.087056461101846, 'n_units_e_l7': 41, 'dropout_l7': 0.4569479505848646, 'n_units_e_l8': 120, 'dropout_l8': 0.13696550966835463, 'n_units_e_l9': 399, 'dropout_l9': 0.030424719619815355, 'n_units_e_l10': 270, 'dropout_l10': 0.07571576467428626, 'n_units_e_l11': 420, 'dropout_l11': 0.13501084404776859, 'n_units_e_l12': 444, 'dropout_l12': 0.0010159304140803793, 'n_units_e_l13': 378, 'dropout_l13': 0.27688932394299237, 'n_units_e_l14': 36, 'dropout_l14': 0.23335480011282855, 'n_units_e_l15': 95, 'dropout_l15': 0.029187000983567968, 'n_units_e_l16': 110, 'dropout_l16': 0.3821666802942647, 'n_units_e_l17': 450, 'dropout_l17': 0.24847995012203922, 'n_units_e_l18': 383, 'dropout_l18': 0.043464041460701375, 'n_units_e_l19': 139, 'dropout_l19': 0.4041576394420242, 'n_units_e_l20': 246, 'dropout_l20': 0.29388853361271394, 'n_units_e_l21': 163, 'dropout_l21': 0.4257647630355981, 'n_units_e_l22': 195, 'dropout_l22': 0.26042053276390853, 'n_units_e_l23': 208, 'dropout_l23': 0.4783466958878107, 'n_units_e_l24': 287, 'dropout_l24': 0.31283121488398047, 'n_units_e_l25': 42, 'dropout_l25': 0.16067637104663812, 'n_units_de_l0': 203, 'n_units_de_l1': 448, 'n_units_de_l2': 261, 'n_units_de_l3': 133, 'n_units_de_l4': 86, 'n_units_de_l5': 166, 'n_units_de_l6': 214, 'n_units_de_l7': 338, 'n_units_de_l8': 319, 'n_units_de_l9': 410, 'n_units_de_l10': 205, 'n_units_de_l11': 88, 'n_units_de_l12': 257, 'n_units_de_l13': 289, 'n_units_de_l14': 100, 'n_units_de_l15': 365, 'n_units_de_l16': 360, 'n_units_de_l17': 247, 'n_units_de_l18': 356, 'n_units_de_l19': 32, 'n_units_de_l20': 443, 'n_units_de_l21': 395, 'n_units_de_l22': 182, 'n_units_de_l23': 328, 'n_units_de_l24': 65, 'n_units_de_l25': 149, 'n_units_de_l26': 140, 'dropout_l26': 0.11531928575857475, 'n_units_de_l27': 472, 'dropout_l27': 0.19565477861574188, 'n_units_de_l28': 385, 'dropout_l28': 0.2160767873690741, 'n_units_de_l29': 66, 'dropout_l29': 0.10791925601308872, 'n_units_di_l0': 244, 'n_units_di_l1': 431, 'n_units_di_l2': 332, 'n_units_di_l3': 226, 'n_units_di_l4': 38, 'n_units_di_l5': 102, 'n_units_di_l6': 379, 'n_units_di_l7': 62, 'n_units_di_l8': 326, 'n_units_di_l9': 257, 'n_units_di_l10': 418, 'n_units_di_l11': 47, 'n_units_di_l12': 368, 'n_units_di_l13': 452, 'n_units_di_l14': 151, 'n_units_di_l15': 212, 'n_units_di_l16': 217, 'n_units_di_l17': 74, 'n_units_di_l18': 61, 'n_units_di_l19': 161, 'n_units_di_l20': 458, 'optimizer': 'Adam', 'lr': 0.00010284383664145362, 'beta1': 0.7965816333040128, 'beta2': 0.9069278102172613, 'lrd': 0.0245377971653829, 'beta1d': 0.7870846209437163, 'beta2d': 0.9955524248846114}.
