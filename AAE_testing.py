import pandas as pd

import utils
from AAE import AAE_archi_opt
import os
import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy, one_hot

from torch import cuda
import itertools
from data import main_u
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

cuda = True if torch.cuda.is_available() else False
torch.cuda.empty_cache()
torch.manual_seed(0)


"""--------------------------------------------------dataset and models--------------------------------------------------"""
in_out = 30
z_dim = 10
label_dim = 4

# dataset = CustomDataset(main_u.X_train_sc.to_numpy(), main_u.y_train.to_numpy())
df = pd.DataFrame(pd.read_csv("/home/silver/PycharmProjects/AAEDRL/AAE/ds.csv"))[:141649]
X_train, X_test, y_train, y_test = main_u.vertical_split(df, main_u.y)
dataset_synth = utils.CustomDataset(X_train.to_numpy(), y_train.to_numpy())
test_loader =utils.dataset_function(dataset_synth, 32, 64, train=False)


encoder_generator = AAE_archi_opt.EncoderGenerator(in_out, z_dim).cuda() if cuda else (
    AAE_archi_opt.EncoderGenerator(in_out, z_dim))

decoder = AAE_archi_opt.Decoder(z_dim+label_dim, in_out, utils.discrete, utils.continuous, utils.binary).cuda() if cuda \
    else (AAE_archi_opt.Decoder(z_dim+label_dim, in_out, utils.discrete, utils.continuous, utils.binary))

discriminator = AAE_archi_opt.Discriminator(z_dim, ).cuda() if cuda else (
    AAE_archi_opt.Discriminator(z_dim, ))

optimizer_G = torch.optim.SGD(itertools.chain(encoder_generator.parameters(), decoder.parameters()),
                              lr=0.000001)
optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=0.001)



# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def test_model(test_loader):
    encoder_generator.eval()
    decoder.eval()
    discriminator.eval()

    total_g_loss = 0
    total_d_loss = 0

    with torch.no_grad():
        for X, y in test_loader:
            valid = torch.ones((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.ones((X.shape[0], 1),
                                                                                                    requires_grad=False)
            fake = torch.zeros((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.zeros((X.shape[0], 1),
                                                                                                     requires_grad=False)
            real = X.type(torch.FloatTensor).cuda() if cuda else X.type(torch.FloatTensor)
            y = y.type(torch.LongTensor).cuda() if cuda else y.type(torch.LongTensor)
            y = one_hot(y, num_classes=4)

            discrete_targets = {}
            continuous_targets = {}
            binary_targets = {}
            for feature, _ in decoder.discrete_features.items():
                discrete_targets[feature] = torch.ones(real.shape[0])

            for feature in decoder.continuous_features:
                continuous_targets[feature] = torch.ones(real.shape[0])

            for feature in decoder.binary_features:
                binary_targets[feature] = torch.ones(real.shape[0])

            encoded = encoder_generator(real)
            dec_input = torch.cat([encoded, y], dim=1)
            discrete_outputs, continuous_outputs, binary_outputs = decoder(dec_input)

            g_loss = (0.1 * binary_cross_entropy(discriminator(encoded), valid) +
                      0.9 * decoder.compute_loss((discrete_outputs, continuous_outputs, binary_outputs),
                                                 (discrete_targets, continuous_targets, binary_targets)))

            z = utils.custom_dist((real.shape[0], z_dim)).cuda() if cuda else utils.custom_dist((real.shape[0], z_dim))
            real_loss = binary_cross_entropy(discriminator(z), valid)
            fake_loss = binary_cross_entropy(discriminator(encoded.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

    avg_g_loss = total_g_loss / len(test_loader)
    avg_d_loss = total_d_loss / len(test_loader)

    return {
        'g_loss': avg_g_loss,
        'd_loss': avg_d_loss
    }
