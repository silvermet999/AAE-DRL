from hyperopt import hp, Trials, tpe, fmin
import numpy as np
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




def train_model(lr, beta1, beta2):
    optimizer_G = torch.optim.Adam(
        itertools.chain(encoder_generator.parameters(), decoder.parameters()), lr=lr, betas=(beta1, beta2)
    )

    def reset_parameters(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    encoder_generator.apply(reset_parameters)
    decoder.apply(reset_parameters)
    discriminator.apply(reset_parameters)

    best_enc_disc = None
    best_params = {}

    for epoch in range(50):
        optim_data_tensor = torch.tensor(main.x_train_rs, dtype=torch.float).cuda() if cuda else torch.tensor(main.x_train_rs, dtype=torch.float)
        optim_real = (optim_data_tensor - optim_data_tensor.mean()) / optim_data_tensor.std()

        optimizer_G.zero_grad()
        optim_encoded = encoder_generator(optim_real)
        optim_enc_disc = discriminator(optim_encoded)

        if torch.isnan(optim_enc_disc).any() or torch.isinf(optim_enc_disc).any():
            print("Warning: NaN or Inf detected in total_enc_disc")


        for i in range (len(optim_enc_disc)):
            if 0.4 < optim_enc_disc[i] < 0.6:
                best_enc_disc = optim_enc_disc
                best_params = {'lr': lr, 'beta1': beta1, 'beta2': beta2}

    return best_params, best_enc_disc


def objective(space):
    lr = space['lr']
    beta1 = space['beta1']
    beta2 = space['beta2']

    best_params, best_enc_disc = train_model(lr, beta1, beta2)
    return -best_enc_disc.mean().item()



space = {
    'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-2)),
    'beta1': hp.uniform('beta1', 0.5, 0.999),
    'beta2': hp.uniform('beta2', 0.9, 0.9999)
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)


