from hyperopt import hp, Trials, tpe, fmin
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR

import main
import itertools
import torch
from torch import nn
import AAE_archi

cuda = True if torch.cuda.is_available() else False
# torch.empty((15000, 15000)).cuda()
# torch.cuda.memory_allocated()


adversarial_loss = nn.BCELoss().cuda() if cuda else nn.BCELoss()
recon_loss = nn.L1Loss().cuda() if cuda else nn.L1Loss()

encoder_generator = AAE_archi.EncoderGenerator().cuda() if cuda else AAE_archi.EncoderGenerator()
decoder = AAE_archi.Decoder().cuda() if cuda else AAE_archi.Decoder()
discriminator = AAE_archi.Discriminator().cuda() if cuda else AAE_archi.Discriminator()


#
# lrg, b1g, b2g, wg, lg, hg, gg
# milestones=[lg, hg], gamma=gg

def train_model(lr, beta1, beta2, weight_decay, low, high, gamma):
    # optimizer_G = torch.optim.Adam(
    #     itertools.chain(encoder_generator.parameters(), decoder.parameters()), lr=lr, betas=(beta1, beta2), weight_decay = weight_decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay = weight_decay)
    # scheduler_G = MultiStepLR(optimizer_G, milestones=[low, high], gamma=gamma)
    scheduler_D = MultiStepLR(optimizer_D, milestones=[low, high], gamma=gamma)

    def reset_parameters(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    encoder_generator.apply(reset_parameters)
    discriminator.apply(reset_parameters)

    best_loss = None
    # best_params = {'lr': lr, 'beta1': beta1, 'beta2': beta2, "weight_decay": weight_decay, "low": low, "high": high,
    #                    "gamma": gamma}
    best_params = {'lr': lr, 'beta1': beta1, 'beta2': beta2, "weight_decay": weight_decay, "low": low, "high": high,
                       "gamma": gamma}

    for epoch in range(10):
        n_batch = len(main.X_train) // 16
        for i in range(n_batch):
            str_idx = i * 16
            end_idx = str_idx + 16
            batch_data = main.X_train.iloc[str_idx:end_idx].values
            train_data_tensor = torch.tensor(batch_data, dtype=torch.float).cuda() if cuda else torch.tensor(batch_data,
                                                                                                             dtype=torch.float)

            real = (train_data_tensor - train_data_tensor.mean()) / train_data_tensor.std()
            valid = torch.ones((train_data_tensor.shape[0], 1)).cuda() if cuda else torch.ones(
                (train_data_tensor.shape[0], 1))
            fake = torch.zeros((train_data_tensor.shape[0], 1)).cuda() if cuda else torch.zeros(
                (train_data_tensor.shape[0], 1))

            # optimizer_G.zero_grad()
            encoded = encoder_generator(real)
            decoded = decoder(encoded)
            # g_loss = 0.01 * adversarial_loss(discriminator(encoded), valid) + 0.99 * recon_loss(decoded, real)
            #
            # g_loss.backward()
            # optimizer_G.step()

            optimizer_D.zero_grad()

            log_normal = torch.distributions.LogNormal(loc=0, scale=1)
            z = log_normal.sample((batch_data.shape[0], AAE_archi.z_dim)).cuda() if cuda else log_normal.sample(
                (batch_data.shape[0], AAE_archi.z_dim))

            # real and fake loss should be close
            # discriminator(z) should be close to 0
            real_loss = adversarial_loss(discriminator(z), valid)
            fake_loss = adversarial_loss(discriminator(encoded.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

        # scheduler_G.step()
        scheduler_D.step()

    # if g_loss < 0.4:
    #     best_lossg = g_loss
    #     best_paramsg.update({'lr': lr, 'beta1': beta1, 'beta2': beta2, 'weight_decay': weight_decay, 'low': low,
    #                         'high': high,
    #                         'gamma': gamma})
    if d_loss < 0.7:
        best_loss = d_loss
        best_params.update({'lr': lr, 'beta1': beta1, 'beta2': beta2, 'weight_decay': weight_decay, 'low': low,
                            'high': high,
                            'gamma': gamma})

    return best_params, best_loss

# 'lrg': lrg, 'beta1g': b1g, 'beta2g': b2g, 'weight_decayg': wg, 'lowg': lg, 'highg': hg,
#                              'gammag': gg


def objective(space):
    lr = space['lr']
    beta1 = space['beta1']
    beta2 = space['beta2']
    weight_decay = space['weight_decay']
    low = space['low']
    high = space['high']
    gamma = space['gamma']
    # lrg = space['lr']
    # b1g = space['beta1']
    # b2g = space['beta2']
    # wg = space['weight_decay']
    # lg = space['low']
    # hg = space['high']
    # gg = space['gamma']

    best_param, best_loss = train_model(lr, beta1, beta2, weight_decay, low, high, gamma)
    return -best_loss.mean().item()



space = {
    'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-2)),
    'beta1': hp.uniform('beta1', 0.5, 0.999),
    'beta2': hp.uniform('beta2', 0.9, 0.9999),
    "weight_decay" : hp.uniform("weight_decay", 0.0001, 0.001),
    "low" : hp.uniform("low", 20, 40),
    "high": hp.uniform("high", 70, 90),
    "gamma" : hp.uniform("gamma", 0, 0.2)
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials)


#  g: {'beta1': 0.9758534841202955, 'beta2': 0.9918395879014463, 'gamma': 0.09240555734722164, 'high': 89.27985628302403, 'low': 28.291395571088543, 'lr': 0.00920828229062108, 'weight_decay': 0.0008904987305687305}
# d: {'beta1': 0.8943741362785895, 'beta2': 0.9995314706647865, 'gamma': 0.017159877576271722, 'high': 88.87749537059518, 'low': 39.93242882049465, 'lr': 1.0846450506796643e-05, 'weight_decay': 0.0004259377371418141}


