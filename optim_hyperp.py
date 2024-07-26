from hyperopt import hp, Trials, tpe, fmin
import numpy as np
import main
import itertools
import torch
from torch import nn
import AAE_archi

cuda = True if torch.cuda.is_available() else False


adversarial_loss = nn.BCELoss().cuda() if cuda else nn.BCELoss()
recon_loss = nn.L1Loss().cuda() if cuda else nn.L1Loss()

encoder_generator = AAE_archi.EncoderGenerator().cuda() if cuda else AAE_archi.EncoderGenerator()
decoder = AAE_archi.Decoder().cuda() if cuda else AAE_archi.Decoder()
discriminator = AAE_archi.Discriminator().cuda() if cuda else AAE_archi.Discriminator()

def train_model(lrG, lrD, beta1G, beta2G, beta1D, beta2D):
    optimizer_G = torch.optim.Adam(
        itertools.chain(encoder_generator.parameters(), decoder.parameters()), lr=lrG, betas=(beta1G, beta2G)
    )
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lrD, betas=(beta1D, beta2D))

    def reset_parameters(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    encoder_generator.apply(reset_parameters)
    decoder.apply(reset_parameters)
    discriminator.apply(reset_parameters)

    best_enc_disc = None
    best_params = {}

    for epoch in range(10):
        optim_data_tensor = torch.tensor(main.x_train_rs, dtype=torch.float).cuda() if cuda else torch.tensor(main.x_train_rs,
                                                                                                       dtype=torch.float)
        optim_real = (optim_data_tensor - optim_data_tensor.mean()) / optim_data_tensor.std()


        optimizer_G.zero_grad()
        optim_encoded = encoder_generator(optim_real)
        total_enc_disc = discriminator(optim_encoded)
        optimizer_D.zero_grad()
        optim_enc_disc_det = discriminator(optim_encoded.detach())


        for i in range(len(total_enc_disc)):
            if 0.4 < total_enc_disc[i].item() < 0.6:
                best_enc_disc = total_enc_disc
                best_params = {'lr': lrG, 'betas': (beta1G, beta2G)}
            if 0.4 < optim_enc_disc_det[i].item() < 0.6:
                best_enc_disc_det = optim_enc_disc_det
                best_params = {'lr': lrD, 'betas': (beta1D, beta2D)}


    return best_params, best_enc_disc, best_enc_disc_det


def objective(space):
    try:
        lrG = np.exp(space['lrG'])
        beta1G = space['beta1G']
        beta2G = space['beta2G']
        lrD = np.exp(space['lrD'])
        beta1D = space['beta1D']
        beta2D = space['beta2D']

        best_params, best_enc_disc, best_enc_disc_det = train_model(lrG, lrD, beta1G, beta2G, beta1D, beta2D)
        return -best_enc_disc.mean().item()
    except Exception as e:
        print(f"Error in objective function: {e}")
        return np.inf


space = {
    'lrG': hp.uniform('lrG', np.log(0.0001), np.log(0.001)),
    'beta1G': hp.uniform('beta1G', 0.5, 0.99),
    'beta2G': hp.uniform('beta2G', 0.9, 0.999),
    'lrD': hp.uniform('lrD', np.log(0.0001), np.log(0.001)),
    'beta1D': hp.uniform('beta1D', 0.5, 0.99),
    'beta2D': hp.uniform('beta2D', 0.9, 0.999)
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)


