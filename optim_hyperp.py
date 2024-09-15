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



def define_model():
    encoder = AAE_archi.encoder_generator
    decoder = AAE_archi.decoder
    discriminator = AAE_archi.discriminator

    return encoder, decoder, discriminator





def objective(trial):
    enc, dec, disc = define_model()
    adversarial_loss = nn.BCELoss().cuda()
    recon_loss = nn.L1Loss().cuda()
    #
    lr = trial.suggest_float('lr', 1e-5, 0.01, log=True)
    beta1 = trial.suggest_float('beta1', 0.5, 0.9)
    beta2 = trial.suggest_float('beta2', 0.9, 0.999)
    lrd = trial.suggest_float('lrd', 1e-5, 0.01, log=True)
    beta1d = trial.suggest_float('beta1d', 0.5, 0.9)
    beta2d = trial.suggest_float('beta2d', 0.9, 0.999)
    optimizer_G = torch.optim.Adam(itertools.chain(enc.parameters(), dec.parameters()), lr=lr, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(disc.parameters(), lr=lrd, betas=(beta1d, beta2d))

    train = main.X_train_rs
    recon_losses = []
    adversarial_losses = []

    epochs = trial.suggest_int("epochs", 100, 300)
    batches = trial.suggest_int("batches", 32, 128)
    for epoch in range(epochs):
        n_batch = len(train) // batches
        for i in range(n_batch):
            str_idx = i * batches
            end_idx = str_idx + batches
            batch_data = train[str_idx:end_idx]

            train_data_tensor = torch.tensor(batch_data, dtype=torch.float).cuda() if cuda else torch.tensor(batch_data, dtype=torch.float)
            real = (train_data_tensor - train_data_tensor.mean()) / train_data_tensor.std()
            valid = torch.ones((train_data_tensor.shape[0], 1)).cuda() if cuda else torch.ones((train_data_tensor.shape[0], 1))
            fake = torch.zeros((train_data_tensor.shape[0], 1)).cuda() if cuda else torch.zeros((train_data_tensor.shape[0], 1))

            optimizer_G.zero_grad()
            encoded = enc(real)
            labels_tensor = AAE_archi.encoded_tensor[str_idx:end_idx]
            dec_input = torch.cat([encoded, labels_tensor], dim=1)
            decoded = dec(dec_input)
            g_loss = 0.001 * adversarial_loss(disc(encoded), valid)+0.999 * recon_loss(decoded, real)

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
    n_batch = len(train)
    for i in range(n_batch):
        str_idx = i * 1
        end_idx = str_idx + 1
        with torch.no_grad():
            val_tensor = torch.tensor(train[str_idx:end_idx], dtype=torch.float).cuda() if cuda else torch.tensor(
                train[str_idx:end_idx], dtype=torch.float)
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
    
    
