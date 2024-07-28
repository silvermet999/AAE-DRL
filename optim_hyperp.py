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




def train_model(lr, beta1, beta2, weight_decay, low, high, gamma):
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay = weight_decay)
    scheduler_D = MultiStepLR(optimizer_D, milestones=[low, high], gamma=gamma)

    def reset_parameters(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    encoder_generator.apply(reset_parameters)
    discriminator.apply(reset_parameters)

    best_loss = None
    best_params = {}

    for epoch in range(10):
        n_batch = len(main.x_train) // 16
        for i in range(n_batch):
            str_idx = i * 16
            end_idx = str_idx + 16
            df_batch = main.x_train.iloc[str_idx:end_idx].values
            optim_data_tensor = torch.tensor(df_batch, dtype=torch.float).cuda() if cuda else torch.tensor(df_batch, dtype=torch.float)
            optim_real = (optim_data_tensor - optim_data_tensor.mean()) / optim_data_tensor.std()
            optim_valid = torch.ones((optim_data_tensor.shape[0], 1)).cuda() if cuda else torch.ones(
                (optim_data_tensor.shape[0], 1))
            optim_fake = torch.zeros((optim_data_tensor.shape[0], 1)).cuda() if cuda else torch.zeros(
                (optim_data_tensor.shape[0], 1))

            optimizer_D.zero_grad()
            optim_encoded = encoder_generator(optim_real)

            log_normal = torch.distributions.LogNormal(loc=0, scale=1)
            optim_z = log_normal.sample((optim_data_tensor.shape[0], AAE_archi.z_dim)).cuda() if cuda else log_normal.sample(
                (optim_data_tensor.shape[0], AAE_archi.z_dim))

            optim_real_loss = adversarial_loss(discriminator(optim_z), optim_valid)
            optim_fake_loss = adversarial_loss(discriminator(optim_encoded.detach()), optim_fake)
            optim_d_loss = 0.5 * (optim_real_loss + optim_fake_loss)

            optim_d_loss.backward()
            optimizer_D.step()


        scheduler_D.step()

    if optim_d_loss <= 0.5:
        best_loss = optim_d_loss
        best_params = {'lr': lr, 'beta1': beta1, 'beta2': beta2, "weight_decay" : weight_decay, "low" : low, "high": high, "gamma" : gamma}

    return best_params, best_loss


def objective(space):
    lr = space['lr']
    beta1 = space['beta1']
    beta2 = space['beta2']
    weight_decay = space['weight_decay']
    low = space['low']
    high = space['high']
    gamma = space['gamma']

    best_params, best_loss = train_model(lr, beta1, beta2, weight_decay, low, high, gamma)
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
            max_evals=10,
            trials=trials)


