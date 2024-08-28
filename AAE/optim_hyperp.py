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
#             p = trial.suggest_float("dropout_l{}".format(i), 0, 0.5)
#             layers_de.append(nn.Dropout(p))
#             in_features_de = out_features
#         layers_de.append(nn.Linear(in_features_de, 105))
#         # layers_de.append(nn.Tanh())
#         self.decoder = nn.Sequential(*layers_de)
#
#     def forward(self, x):
#         return self.decoder(x)


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
    # n_layers_e = trial.suggest_int("n_layers_e", 20, 40)
    # n_layers_de = trial.suggest_int("n_layers_de", 20, 40)
    n_layers_di = trial.suggest_int("n_layers_di", 10, 30)
    # in_features_e = 105
    # in_features_de = 32
    in_features_di = 32

    # encoder = Encoder(trial, n_layers_e, in_features_e).cuda() if cuda else Encoder(trial, n_layers_e, in_features_e)
    # decoder = Decoder(trial, n_layers_de, in_features_de).cuda() if cuda else Decoder(trial, n_layers_de, in_features_de)
    encoder = AAE_archi.encoder_generator
    decoder = AAE_archi.decoder
    discriminator = Discriminator(trial, n_layers_di, in_features_di).cuda() if cuda else Discriminator(trial, n_layers_di, in_features_di)

    return encoder, decoder, discriminator





def objective(trial):
    enc, dec, disc = define_model(trial)
    adversarial_loss = nn.BCELoss()
    recon_loss = nn.L1Loss()

    # lr = trial.suggest_float('lr', 1e-5, 0.1, log=True)
    # beta1 = trial.suggest_float('beta1', 0.5, 0.9)
    # beta2 = trial.suggest_float('beta2', 0.9, 0.999)
    lrd = trial.suggest_float('lrd', 1e-5, 0.1, log=True)
    beta1d = trial.suggest_float('beta1d', 0.5, 0.9)
    beta2d = trial.suggest_float('beta2d', 0.9, 0.999)
    # 'lr': 6.0798214947201093e-05, 'beta1': 0.5596538426477291, 'beta2': 0.9657272988779723
    optimizer_G = torch.optim.Adam(itertools.chain(enc.parameters(), dec.parameters()), lr=6.0798214947201093e-05,
                                                 betas=(.5596538426477291, .9657272988779723))
    optimizer_D = torch.optim.Adam(disc.parameters(), lr=lrd, betas=(beta1d, beta2d))

    train = main.X_train_rs
    # recon = []
    adv = []


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
            mu, logvar, encoded = enc(real)
            decoded = dec(encoded)
            g_loss = (0.001 * adversarial_loss(disc(encoded), valid) + 0.999 * recon_loss(decoded, real))

            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            z = torch.normal(0, 1, (batch_data.shape[0], 32)).cuda() if cuda else torch.normal(0, 1, (batch_data.shape[0], 32))

            # real and fake loss should be close
            # discriminator(z) should be close to 0
            real_loss = adversarial_loss(disc(z), valid)
            fake_loss = adversarial_loss(disc(encoded.detach().sigmoid()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

        # recon.append(np.mean(g_loss.item()))
        adv.append(np.mean(d_loss.item()))
    # recon_mean = np.mean(recon)
    adv_mean = np.mean(adv)

    return adv_mean



study = optuna.create_study(directions=["minimize"])
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

# [I 2024-08-10 15:40:57,666] Trial 0 finished with values: [0.09702626450194253, 0.039081158461545716] and parameters: {'n_layers_e': 44, 'n_layers_de': 37, 'n_layers_di': 11, 'n_units_e_l0': 369, 'dropout_l0': 0.2912014074950425, 'n_units_e_l1': 264, 'dropout_l1': 0.35957829905370775, 'n_units_e_l2': 143, 'dropout_l2': 0.16634929998161474, 'n_units_e_l3': 418, 'dropout_l3': 0.14488264346896107, 'n_units_e_l4': 483, 'dropout_l4': 0.34961088800051693, 'n_units_e_l5': 340, 'dropout_l5': 0.2882157846404448, 'n_units_e_l6': 374, 'dropout_l6': 0.02374574253753159, 'n_units_e_l7': 185, 'dropout_l7': 0.2984299208956127, 'n_units_e_l8': 152, 'dropout_l8': 0.07317829255139408, 'n_units_e_l9': 225, 'dropout_l9': 0.4633390011826975, 'n_units_e_l10': 248, 'dropout_l10': 0.18925529370010746, 'n_units_e_l11': 406, 'dropout_l11': 0.04124974957292793, 'n_units_e_l12': 501, 'dropout_l12': 0.3050765260201068, 'n_units_e_l13': 252, 'dropout_l13': 0.4529985369856203, 'n_units_e_l14': 392, 'dropout_l14': 0.09256666951563924, 'n_units_e_l15': 313, 'dropout_l15': 0.0034110266440671166, 'n_units_e_l16': 166, 'dropout_l16': 0.2433992663000255, 'n_units_e_l17': 54, 'dropout_l17': 0.3637684362506816, 'n_units_e_l18': 142, 'dropout_l18': 0.11901943868375703, 'n_units_e_l19': 224, 'dropout_l19': 0.12485899981978027, 'n_units_e_l20': 321, 'dropout_l20': 0.40862458111926414, 'n_units_e_l21': 493, 'dropout_l21': 0.25096111389024267, 'n_units_e_l22': 303, 'dropout_l22': 0.17344412654847136, 'n_units_e_l23': 153, 'dropout_l23': 0.19765134545072033, 'n_units_e_l24': 138, 'dropout_l24': 0.0293149257260214, 'n_units_e_l25': 252, 'dropout_l25': 0.43794389274878254, 'n_units_e_l26': 110, 'dropout_l26': 0.4365971670818343, 'n_units_e_l27': 376, 'dropout_l27': 0.4932216553741478, 'n_units_e_l28': 245, 'dropout_l28': 0.17462278881475202, 'n_units_e_l29': 349, 'dropout_l29': 0.04790680468013431, 'n_units_e_l30': 478, 'dropout_l30': 0.189794748643973, 'n_units_e_l31': 424, 'dropout_l31': 0.28708515210653035, 'n_units_e_l32': 63, 'dropout_l32': 0.12792539876826858, 'n_units_e_l33': 204, 'dropout_l33': 0.04126183161713565, 'n_units_e_l34': 438, 'dropout_l34': 0.46693733936443116, 'n_units_e_l35': 390, 'dropout_l35': 0.16244944823482776, 'n_units_e_l36': 456, 'dropout_l36': 0.3438666329016294, 'n_units_e_l37': 294, 'dropout_l37': 0.3571553789563026, 'n_units_e_l38': 348, 'dropout_l38': 0.30372651918722593, 'n_units_e_l39': 239, 'dropout_l39': 0.3367211118052325, 'n_units_e_l40': 479, 'dropout_l40': 0.44870344468741424, 'n_units_e_l41': 260, 'dropout_l41': 0.00375152328986611, 'n_units_e_l42': 457, 'dropout_l42': 0.34571830645315565, 'n_units_e_l43': 310, 'dropout_l43': 0.3001775340913507, 'n_units_de_l0': 225, 'n_units_de_l1': 281, 'n_units_de_l2': 303, 'n_units_de_l3': 444, 'n_units_de_l4': 59, 'n_units_de_l5': 124, 'n_units_de_l6': 245, 'n_units_de_l7': 293, 'n_units_de_l8': 442, 'n_units_de_l9': 444, 'n_units_de_l10': 494, 'n_units_de_l11': 438, 'n_units_de_l12': 414, 'n_units_de_l13': 83, 'n_units_de_l14': 255, 'n_units_de_l15': 420, 'n_units_de_l16': 315, 'n_units_de_l17': 473, 'n_units_de_l18': 52, 'n_units_de_l19': 161, 'n_units_de_l20': 501, 'n_units_de_l21': 88, 'n_units_de_l22': 439, 'n_units_de_l23': 441, 'n_units_de_l24': 257, 'n_units_de_l25': 313, 'n_units_de_l26': 472, 'n_units_de_l27': 326, 'n_units_de_l28': 382, 'n_units_de_l29': 187, 'n_units_de_l30': 188, 'n_units_de_l31': 248, 'n_units_de_l32': 98, 'n_units_de_l33': 158, 'n_units_de_l34': 35, 'n_units_de_l35': 236, 'n_units_de_l36': 233, 'n_units_di_l0': 493, 'n_units_di_l1': 445, 'n_units_di_l2': 64, 'n_units_di_l3': 290, 'n_units_di_l4': 359, 'n_units_di_l5': 421, 'n_units_di_l6': 403, 'n_units_di_l7': 118, 'n_units_di_l8': 162, 'n_units_di_l9': 134, 'n_units_di_l10': 466, 'optimizer': 'Adam', 'lr': 1.802133395290409e-05, 'beta1': 0.8077587688567314, 'beta2': 0.980815082090427, 'lrd': 0.001290574059129973, 'beta1d': 0.8339287336360124, 'beta2d': 0.9128002537599588, 'n_ep': 54, 'batch': 99}.
# /home/silver/PycharmProjects/AAEDRL/.venv/bin/python /home/silver/pycharm-community-2024.1.4/plugins/python-ce/helpers/pydev/pydevconsole.py --mode=client --host=127.0.0.1 --port=33207
#
# import sys; print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend(['/home/silver/PycharmProjects/AAEDRL'])
#
# PyDev console: starting.
#
# Python 3.11.2 (main, May  2 2024, 11:59:08) [GCC 12.2.0] on linux
# >>> runfile('/home/silver/PycharmProjects/AAEDRL/AAE/optim_hyperp.py', wdir='/home/silver/PycharmProjects/AAEDRL/AAE')
# Backend TkAgg is interactive backend. Turning interactive mode on.
# [I 2024-08-22 21:43:16,171] A new study created in memory with name: no-name-bab43960-b0d3-47f8-b07e-a9e66d8a4aec
# [I 2024-08-22 22:20:52,898] Trial 0 finished with value: 0.04867893885821104 and parameters: {'n_layers_e': 26, 'n_layers_de': 21, 'n_units_e_l0': 291, 'dropout_l0': 0.11869008882653087, 'n_units_e_l1': 350, 'dropout_l1': 0.2244600208536538, 'n_units_e_l2': 211, 'dropout_l2': 0.3759205785067355, 'n_units_e_l3': 252, 'dropout_l3': 0.36823113129866636, 'n_units_e_l4': 131, 'dropout_l4': 0.4976431121132785, 'n_units_e_l5': 200, 'dropout_l5': 0.054043832415111404, 'n_units_e_l6': 125, 'dropout_l6': 0.32426135089738095, 'n_units_e_l7': 482, 'dropout_l7': 0.004072887711605577, 'n_units_e_l8': 358, 'dropout_l8': 0.1275449214975269, 'n_units_e_l9': 497, 'dropout_l9': 0.3895469060240036, 'n_units_e_l10': 507, 'dropout_l10': 0.4000784132471235, 'n_units_e_l11': 129, 'dropout_l11': 0.15450535177563135, 'n_units_e_l12': 362, 'dropout_l12': 0.1879448807460054, 'n_units_e_l13': 391, 'dropout_l13': 0.31660484775080355, 'n_units_e_l14': 452, 'dropout_l14': 0.48504643080261145, 'n_units_e_l15': 320, 'dropout_l15': 0.2700967686150639, 'n_units_e_l16': 36, 'dropout_l16': 0.24205793583080354, 'n_units_e_l17': 157, 'dropout_l17': 0.16317148147048893, 'n_units_e_l18': 293, 'dropout_l18': 0.25143542783073497, 'n_units_e_l19': 235, 'dropout_l19': 0.10542930792051308, 'n_units_e_l20': 290, 'dropout_l20': 0.4556656361493996, 'n_units_e_l21': 200, 'dropout_l21': 0.2332258974758853, 'n_units_e_l22': 75, 'dropout_l22': 0.21708148449977654, 'n_units_e_l23': 187, 'dropout_l23': 0.3225685184911873, 'n_units_e_l24': 172, 'dropout_l24': 0.26451380659772095, 'n_units_e_l25': 338, 'dropout_l25': 0.015300282621798555, 'n_units_de_l0': 394, 'n_units_de_l1': 240, 'n_units_de_l2': 232, 'n_units_de_l3': 272, 'n_units_de_l4': 132, 'n_units_de_l5': 230, 'n_units_de_l6': 404, 'n_units_de_l7': 289, 'n_units_de_l8': 266, 'n_units_de_l9': 294, 'n_units_de_l10': 359, 'n_units_de_l11': 271, 'n_units_de_l12': 240, 'n_units_de_l13': 358, 'n_units_de_l14': 317, 'n_units_de_l15': 461, 'n_units_de_l16': 409, 'n_units_de_l17': 411, 'n_units_de_l18': 72, 'n_units_de_l19': 280, 'n_units_de_l20': 279, 'optimizer': 'Adam', 'lr': 6.0798214947201093e-05, 'beta1': 0.5596538426477291, 'beta2': 0.9657272988779723}. Best is trial 0 with value: 0.04867893885821104.
# /home/silver/PycharmProjects/AAEDRL/.venv/bin/python /home/silver/pycharm-community-2024.1.4/plugins/python-ce/helpers/pydev/pydevconsole.py --mode=client --host=127.0.0.1 --port=46847
#
# import sys; print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend(['/home/silver/PycharmProjects/AAEDRL'])
#
# PyDev console: starting.
#
# Python 3.11.2 (main, May  2 2024, 11:59:08) [GCC 12.2.0] on linux
# >>> runfile('/home/silver/PycharmProjects/AAEDRL/AAE/optim_hyperp.py', wdir='/home/silver/PycharmProjects/AAEDRL/AAE')
# Backend TkAgg is interactive backend. Turning interactive mode on.
# [I 2024-08-22 22:45:13,445] A new study created in memory with name: no-name-e02c57fa-bfaa-4f50-a096-2e1a5ab99827
# [I 2024-08-22 23:01:44,695] Trial 0 finished with value: 0.6957516431808471 and parameters: {'n_layers_e': 26, 'n_layers_di': 10, 'n_units_e_l0': 377, 'dropout_l0': 0.46841630015425284, 'n_units_e_l1': 324, 'dropout_l1': 0.07071412780217767, 'n_units_e_l2': 435, 'dropout_l2': 0.06054289785963346, 'n_units_e_l3': 85, 'dropout_l3': 0.28011912634602065, 'n_units_e_l4': 425, 'dropout_l4': 0.2604051742032295, 'n_units_e_l5': 273, 'dropout_l5': 0.35401664264402044, 'n_units_e_l6': 379, 'dropout_l6': 0.24244641230352898, 'n_units_e_l7': 331, 'dropout_l7': 0.17380471785789248, 'n_units_e_l8': 382, 'dropout_l8': 0.039827166588156326, 'n_units_e_l9': 511, 'dropout_l9': 0.009180305453471815, 'n_units_e_l10': 430, 'dropout_l10': 0.29537452106166345, 'n_units_e_l11': 450, 'dropout_l11': 0.430531772932247, 'n_units_e_l12': 322, 'dropout_l12': 0.057611912391179354, 'n_units_e_l13': 100, 'dropout_l13': 0.24621249291655495, 'n_units_e_l14': 73, 'dropout_l14': 0.33888939863225476, 'n_units_e_l15': 390, 'dropout_l15': 0.18720829671915967, 'n_units_e_l16': 204, 'dropout_l16': 0.17619488672600464, 'n_units_e_l17': 249, 'dropout_l17': 0.3313897391916054, 'n_units_e_l18': 205, 'dropout_l18': 0.3484344580282585, 'n_units_e_l19': 221, 'dropout_l19': 0.17111555905377762, 'n_units_e_l20': 338, 'dropout_l20': 0.3906622455210867, 'n_units_e_l21': 373, 'dropout_l21': 0.2875325057717947, 'n_units_e_l22': 376, 'dropout_l22': 0.32569954279160807, 'n_units_e_l23': 411, 'dropout_l23': 0.47726998006105525, 'n_units_e_l24': 137, 'dropout_l24': 0.2477287264156281, 'n_units_e_l25': 334, 'dropout_l25': 0.36545794850508445, 'n_units_di_l0': 100, 'n_units_di_l1': 102, 'n_units_di_l2': 270, 'n_units_di_l3': 161, 'n_units_di_l4': 450, 'n_units_di_l5': 440, 'n_units_di_l6': 462, 'n_units_di_l7': 347, 'n_units_di_l8': 123, 'n_units_di_l9': 465, 'optimizer': 'Adam', 'lrd': 0.06330916965374074, 'beta1d': 0.8053969751069527, 'beta2d': 0.9880762969430811}. Best is trial 0 with value: 0.6957516431808471.
# /home/silver/PycharmProjects/AAEDRL/.venv/bin/python /home/silver/pycharm-community-2024.1.4/plugins/python-ce/helpers/pydev/pydevconsole.py --mode=client --host=127.0.0.1 --port=46847
#
# import sys; print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend(['/home/silver/PycharmProjects/AAEDRL'])
#
# PyDev console: starting.
#
# Python 3.11.2 (main, May  2 2024, 11:59:08) [GCC 12.2.0] on linux
# >>> runfile('/home/silver/PycharmProjects/AAEDRL/AAE/optim_hyperp.py', wdir='/home/silver/PycharmProjects/AAEDRL/AAE')
# Backend TkAgg is interactive backend. Turning interactive mode on.
# [I 2024-08-22 22:45:13,445] A new study created in memory with name: no-name-e02c57fa-bfaa-4f50-a096-2e1a5ab99827
# [I 2024-08-22 23:01:44,695] Trial 0 finished with value: 0.6957516431808471 and parameters: {'n_layers_e': 26, 'n_layers_di': 10, 'n_units_e_l0': 377, 'dropout_l0': 0.46841630015425284, 'n_units_e_l1': 324, 'dropout_l1': 0.07071412780217767, 'n_units_e_l2': 435, 'dropout_l2': 0.06054289785963346, 'n_units_e_l3': 85, 'dropout_l3': 0.28011912634602065, 'n_units_e_l4': 425, 'dropout_l4': 0.2604051742032295, 'n_units_e_l5': 273, 'dropout_l5': 0.35401664264402044, 'n_units_e_l6': 379, 'dropout_l6': 0.24244641230352898, 'n_units_e_l7': 331, 'dropout_l7': 0.17380471785789248, 'n_units_e_l8': 382, 'dropout_l8': 0.039827166588156326, 'n_units_e_l9': 511, 'dropout_l9': 0.009180305453471815, 'n_units_e_l10': 430, 'dropout_l10': 0.29537452106166345, 'n_units_e_l11': 450, 'dropout_l11': 0.430531772932247, 'n_units_e_l12': 322, 'dropout_l12': 0.057611912391179354, 'n_units_e_l13': 100, 'dropout_l13': 0.24621249291655495, 'n_units_e_l14': 73, 'dropout_l14': 0.33888939863225476, 'n_units_e_l15': 390, 'dropout_l15': 0.18720829671915967, 'n_units_e_l16': 204, 'dropout_l16': 0.17619488672600464, 'n_units_e_l17': 249, 'dropout_l17': 0.3313897391916054, 'n_units_e_l18': 205, 'dropout_l18': 0.3484344580282585, 'n_units_e_l19': 221, 'dropout_l19': 0.17111555905377762, 'n_units_e_l20': 338, 'dropout_l20': 0.3906622455210867, 'n_units_e_l21': 373, 'dropout_l21': 0.2875325057717947, 'n_units_e_l22': 376, 'dropout_l22': 0.32569954279160807, 'n_units_e_l23': 411, 'dropout_l23': 0.47726998006105525, 'n_units_e_l24': 137, 'dropout_l24': 0.2477287264156281, 'n_units_e_l25': 334, 'dropout_l25': 0.36545794850508445, 'n_units_di_l0': 100, 'n_units_di_l1': 102, 'n_units_di_l2': 270, 'n_units_di_l3': 161, 'n_units_di_l4': 450, 'n_units_di_l5': 440, 'n_units_di_l6': 462, 'n_units_di_l7': 347, 'n_units_di_l8': 123, 'n_units_di_l9': 465, 'optimizer': 'Adam', 'lrd': 0.06330916965374074, 'beta1d': 0.8053969751069527, 'beta2d': 0.9880762969430811}. Best is trial 0 with value: 0.6957516431808471.
# [I 2024-08-22 23:21:41,799] Trial 1 finished with value: 0.6944085687398911 and parameters: {'n_layers_e': 23, 'n_layers_di': 14, 'n_units_e_l0': 82, 'dropout_l0': 0.3642940046913151, 'n_units_e_l1': 282, 'dropout_l1': 0.1704116839359287, 'n_units_e_l2': 212, 'dropout_l2': 0.23081211890336256, 'n_units_e_l3': 425, 'dropout_l3': 0.39835418002093953, 'n_units_e_l4': 256, 'dropout_l4': 0.1207924643132432, 'n_units_e_l5': 273, 'dropout_l5': 0.4453249333630154, 'n_units_e_l6': 183, 'dropout_l6': 0.11523387429263127, 'n_units_e_l7': 277, 'dropout_l7': 0.39164090087378933, 'n_units_e_l8': 447, 'dropout_l8': 0.3780921609053698, 'n_units_e_l9': 460, 'dropout_l9': 0.17300760197696302, 'n_units_e_l10': 181, 'dropout_l10': 0.2197984368216918, 'n_units_e_l11': 153, 'dropout_l11': 0.3104873218608736, 'n_units_e_l12': 234, 'dropout_l12': 0.41238919088812337, 'n_units_e_l13': 402, 'dropout_l13': 0.014162744949949735, 'n_units_e_l14': 364, 'dropout_l14': 0.2753296283817653, 'n_units_e_l15': 151, 'dropout_l15': 0.2804902438738813, 'n_units_e_l16': 51, 'dropout_l16': 0.23455520584120415, 'n_units_e_l17': 123, 'dropout_l17': 0.27036103878753964, 'n_units_e_l18': 317, 'dropout_l18': 0.19349062183150012, 'n_units_e_l19': 193, 'dropout_l19': 0.07066041674835305, 'n_units_e_l20': 435, 'dropout_l20': 0.4377840702970794, 'n_units_e_l21': 149, 'dropout_l21': 0.1532419172796533, 'n_units_e_l22': 36, 'dropout_l22': 0.37997323887389184, 'n_units_di_l0': 275, 'n_units_di_l1': 240, 'n_units_di_l2': 132, 'n_units_di_l3': 219, 'n_units_di_l4': 242, 'n_units_di_l5': 478, 'n_units_di_l6': 129, 'n_units_di_l7': 390, 'n_units_di_l8': 344, 'n_units_di_l9': 188, 'n_units_di_l10': 481, 'n_units_di_l11': 34, 'n_units_di_l12': 285, 'n_units_di_l13': 89, 'optimizer': 'Adam', 'lrd': 0.007604407156891325, 'beta1d': 0.6088246849505203, 'beta2d': 0.969038864389304}. Best is trial 1 with value: 0.6944085687398911.

# [I 2024-08-22 23:34:33,952] A new study created in memory with name: no-name-93e25a00-67ec-47dc-8406-984106c38787
# [I 2024-08-23 01:06:45,293] Trial 0 finished with value: 0.7335322690010071 and parameters: {'n_layers_di': 28, 'n_units_di_l0': 137, 'dropout_l0': 0.34146481054887756, 'n_units_di_l1': 142, 'dropout_l1': 0.3541134367750153, 'n_units_di_l2': 289, 'dropout_l2': 0.41400707804378145, 'n_units_di_l3': 372, 'dropout_l3': 0.24389440421004305, 'n_units_di_l4': 171, 'dropout_l4': 0.3922285396632153, 'n_units_di_l5': 179, 'dropout_l5': 0.0902228286658554, 'n_units_di_l6': 368, 'dropout_l6': 0.18018658728378806, 'n_units_di_l7': 456, 'dropout_l7': 0.24105409138651213, 'n_units_di_l8': 430, 'dropout_l8': 0.20522903646672158, 'n_units_di_l9': 65, 'dropout_l9': 0.43151151606091015, 'n_units_di_l10': 319, 'dropout_l10': 0.14311999348557508, 'n_units_di_l11': 425, 'dropout_l11': 0.2289870907331999, 'n_units_di_l12': 168, 'dropout_l12': 0.3279297251565533, 'n_units_di_l13': 189, 'dropout_l13': 0.16177300735066913, 'n_units_di_l14': 171, 'dropout_l14': 0.195132799431311, 'n_units_di_l15': 405, 'dropout_l15': 0.0009450975330277944, 'n_units_di_l16': 194, 'dropout_l16': 0.11852158882393338, 'n_units_di_l17': 492, 'dropout_l17': 0.00664864526970238, 'n_units_di_l18': 54, 'dropout_l18': 0.10017724650822973, 'n_units_di_l19': 50, 'dropout_l19': 0.21906724297614177, 'n_units_di_l20': 73, 'dropout_l20': 0.27600514321899233, 'n_units_di_l21': 47, 'dropout_l21': 0.08312377861177311, 'n_units_di_l22': 98, 'dropout_l22': 0.2775717439769265, 'n_units_di_l23': 264, 'dropout_l23': 0.40248971350900486, 'n_units_di_l24': 172, 'dropout_l24': 0.28372167105190954, 'n_units_di_l25': 421, 'dropout_l25': 0.3421972428477292, 'n_units_di_l26': 85, 'dropout_l26': 0.39930037395642837, 'n_units_di_l27': 329, 'dropout_l27': 0.3282931992903736, 'lrd': 0.048145284536335736, 'beta1d': 0.7514464237079014, 'beta2d': 0.9725088153894389}. Best is trial 0 with value: 0.7335322690010071.
# [I 2024-08-23 02:39:38,536] Trial 1 finished with value: 0.6935158807039261 and parameters: {'n_layers_di': 28, 'n_units_di_l0': 201, 'dropout_l0': 0.04104446508483589, 'n_units_di_l1': 309, 'dropout_l1': 0.13638471710881284, 'n_units_di_l2': 284, 'dropout_l2': 0.433775451450192, 'n_units_di_l3': 66, 'dropout_l3': 0.09578564922267335, 'n_units_di_l4': 176, 'dropout_l4': 0.2826452099138761, 'n_units_di_l5': 93, 'dropout_l5': 0.4140934888490146, 'n_units_di_l6': 225, 'dropout_l6': 0.006235755606376847, 'n_units_di_l7': 495, 'dropout_l7': 0.2666453966255024, 'n_units_di_l8': 456, 'dropout_l8': 0.2902363617947019, 'n_units_di_l9': 447, 'dropout_l9': 0.36955640078826457, 'n_units_di_l10': 238, 'dropout_l10': 0.33252623279450944, 'n_units_di_l11': 511, 'dropout_l11': 0.4971697080354652, 'n_units_di_l12': 368, 'dropout_l12': 0.08474843941864718, 'n_units_di_l13': 326, 'dropout_l13': 0.004826946018196054, 'n_units_di_l14': 485, 'dropout_l14': 0.18560779425240242, 'n_units_di_l15': 140, 'dropout_l15': 0.26324323885732, 'n_units_di_l16': 69, 'dropout_l16': 0.17751169736270578, 'n_units_di_l17': 155, 'dropout_l17': 0.18582673782206838, 'n_units_di_l18': 472, 'dropout_l18': 0.13050808507715939, 'n_units_di_l19': 71, 'dropout_l19': 0.250038989370521, 'n_units_di_l20': 354, 'dropout_l20': 0.00896265042480654, 'n_units_di_l21': 90, 'dropout_l21': 0.024001597883306358, 'n_units_di_l22': 358, 'dropout_l22': 0.07822461526957614, 'n_units_di_l23': 245, 'dropout_l23': 0.4201200870016977, 'n_units_di_l24': 409, 'dropout_l24': 0.01586928193817555, 'n_units_di_l25': 135, 'dropout_l25': 0.057345533791874514, 'n_units_di_l26': 512, 'dropout_l26': 0.06299003599459585, 'n_units_di_l27': 147, 'dropout_l27': 0.3944843530872543, 'lrd': 8.81782846132046e-05, 'beta1d': 0.7550380003374881, 'beta2d': 0.9573989879160437}. Best is trial 1 with value: 0.6935158807039261.
# [I 2024-08-23 04:10:52,340] Trial 2 finished with value: 0.6940011477470398 and parameters: {'n_layers_di': 30, 'n_units_di_l0': 119, 'dropout_l0': 0.054814173042958436, 'n_units_di_l1': 215, 'dropout_l1': 0.11729590104187138, 'n_units_di_l2': 222, 'dropout_l2': 0.01203422399796994, 'n_units_di_l3': 372, 'dropout_l3': 0.24090432689768448, 'n_units_di_l4': 279, 'dropout_l4': 0.4391343223644175, 'n_units_di_l5': 220, 'dropout_l5': 0.17466318685015148, 'n_units_di_l6': 249, 'dropout_l6': 0.34191105590804205, 'n_units_di_l7': 203, 'dropout_l7': 0.20927817861023806, 'n_units_di_l8': 398, 'dropout_l8': 0.43364693925410147, 'n_units_di_l9': 88, 'dropout_l9': 0.4641916208076488, 'n_units_di_l10': 99, 'dropout_l10': 0.052867878389609546, 'n_units_di_l11': 426, 'dropout_l11': 0.03710441469708908, 'n_units_di_l12': 74, 'dropout_l12': 0.04206558691762313, 'n_units_di_l13': 163, 'dropout_l13': 0.036415164391322086, 'n_units_di_l14': 155, 'dropout_l14': 0.45841885060655285, 'n_units_di_l15': 510, 'dropout_l15': 0.39026496478961686, 'n_units_di_l16': 398, 'dropout_l16': 0.11573883606278612, 'n_units_di_l17': 399, 'dropout_l17': 0.07895354946642541, 'n_units_di_l18': 478, 'dropout_l18': 0.20680292253368987, 'n_units_di_l19': 111, 'dropout_l19': 0.16029866679150584, 'n_units_di_l20': 114, 'dropout_l20': 0.2616951259914552, 'n_units_di_l21': 59, 'dropout_l21': 0.23675432676060865, 'n_units_di_l22': 165, 'dropout_l22': 0.3658764006614752, 'n_units_di_l23': 268, 'dropout_l23': 0.042917033010746064, 'n_units_di_l24': 50, 'dropout_l24': 0.14432479800687859, 'n_units_di_l25': 421, 'dropout_l25': 0.42938959383727493, 'n_units_di_l26': 140, 'dropout_l26': 0.22701647676192732, 'n_units_di_l27': 458, 'dropout_l27': 0.29316096245112605, 'n_units_di_l28': 77, 'dropout_l28': 0.4021724395725551, 'n_units_di_l29': 331, 'dropout_l29': 0.22311674510105817, 'lrd': 0.0031079110766642003, 'beta1d': 0.5837376222564348, 'beta2d': 0.990979449948866}. Best is trial 1 with value: 0.6935158807039261.
# Trial 6 finished with value: 0.6930056875944137 and parameters: {'n_layers_di': 18, 'n_units_di_l0': 37, 'dropout_l0': 0.3953108842666025, 'n_units_di_l1': 446, 'dropout_l1': 0.46275164457331813, 'n_units_di_l2': 298, 'dropout_l2': 0.41439725774646496, 'n_units_di_l3': 336, 'dropout_l3': 0.38235687521714845, 'n_units_di_l4': 395, 'dropout_l4': 0.4162961310002777, 'n_units_di_l5': 263, 'dropout_l5': 0.40417908930241075, 'n_units_di_l6': 197, 'dropout_l6': 0.3063286162663691, 'n_units_di_l7': 302, 'dropout_l7': 0.45484754782914616, 'n_units_di_l8': 497, 'dropout_l8': 0.23592196910697666, 'n_units_di_l9': 278, 'dropout_l9': 0.33283213596440575, 'n_units_di_l10': 320, 'dropout_l10': 0.38006464784065536, 'n_units_di_l11': 502, 'dropout_l11': 0.15037293254215617, 'n_units_di_l12': 81, 'dropout_l12': 0.4906454386348957, 'n_units_di_l13': 407, 'dropout_l13': 0.4439607716761734, 'n_units_di_l14': 302, 'dropout_l14': 0.18318849312157814, 'n_units_di_l15': 322, 'dropout_l15': 0.345179737062177, 'n_units_di_l16': 286, 'dropout_l16': 0.16306932470098257, 'n_units_di_l17': 65, 'dropout_l17': 0.4407439993652657, 'lrd': 0.0003537881190733828, 'beta1d': 0.7986571988762406, 'beta2d': 0.9520849581646436}. Best is trial 6 with value: 0.6930056875944137.
