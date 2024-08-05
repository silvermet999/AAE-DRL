import torch
import torch.nn as nn
import optuna
import itertools
import main

def define_model(trial):
    n_layers_e = trial.suggest_int("n_layers_e", 20, 50)
    n_layers_de = trial.suggest_int("n_layers_de", 20, 50)
    n_layers_di = trial.suggest_int("n_layers_di", 10, 40)
    
    layers_e = []
    layers_de = []
    layers_di = []

    in_features_e = 105
    in_features_de = 32
    in_features_di = 32

    for i in range(n_layers_e):
        out_features = trial.suggest_int(f"n_units_e_l{i}", 32, 512)
        layers_e.append(nn.Linear(in_features_e, out_features))
        layers_e.append(nn.LeakyReLU())
        layers_e.append(nn.BatchNorm1d(out_features))
        p_e = trial.suggest_float("dropout_l{}".format(i), 0, 0.3)
        layers_e.append(nn.Dropout(p_e))
        in_features_e = out_features
    layers_e.append(nn.Linear(in_features_e, 32))
    encoder = nn.Sequential(*layers_e)

    for i in range(n_layers_de):
        out_features = trial.suggest_int(f"n_units_de_l{i}", 32, 512)
        layers_de.append(nn.Linear(in_features_de, out_features))
        layers_de.append(nn.LeakyReLU())
        layers_de.append(nn.BatchNorm1d(out_features))
        p_de = trial.suggest_float("dropout_l{}".format(i), 0, 0.3)
        layers_de.append(nn.Dropout(p_de))
        in_features_de = out_features
    layers_de.append(nn.Linear(in_features_de, 105))
    layers_de.append(nn.Tanh())
    decoder = nn.Sequential(*layers_de)

    for i in range(n_layers_di):
        out_features = trial.suggest_int(f"n_units_di_l{i}", 32, 128)
        layers_di.append(nn.Linear(in_features_di, out_features))
        layers_di.append(nn.LeakyReLU())
        layers_di.append(nn.BatchNorm1d(out_features))
        p_di = trial.suggest_float("dropout_l{}".format(i), 0, 0.5)
        layers_di.append(nn.Dropout(p_di))
        in_features_di = out_features
    layers_di.append(nn.Linear(in_features_di, 1))
    layers_di.append(nn.Sigmoid())
    discriminator = nn.Sequential(*layers_di)

    return encoder, decoder, discriminator


import torch.optim as optim
from sklearn.model_selection import KFold
import numpy as np

def objective(trial):
    enc, dec, disc = define_model(trial)
    
    
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    beta1 = trial.suggest_float('beta1', 0.5, 0.9)
    beta2 = trial.suggest_float('beta2', 0.9, 0.999)
    lrd = trial.suggest_float('lrd', 1e-5, 1e-1, log=True)
    beta1d = trial.suggest_float('beta1d', 0.5, 0.9)
    beta2d = trial.suggest_float('beta2d', 0.9, 0.999)
    optimizer_G = getattr(optim, optimizer_name)(itertools.chain(enc.parameters(), dec.parameters()), lr=lr, betas=(beta1, beta2))
    optimizer_D = getattr(optim, optimizer_name)(disc.parameters(), lr=lrd, betas=(beta1d, beta2d))
    
    train = main.X_train_rs_cl
    test = main.X_test_rs_cl
    
    adversarial_loss = nn.BCELoss()
    recon_loss = nn.L1Loss()
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    recon_losses = []
    adversarial_losses = []
    
    for epoch in range(100):
        n_batch = len(train) // 32
        for i in range(n_batch):
            str_idx = i * 32
            end_idx = str_idx + 32
            batch_data = train[str_idx:end_idx]
            train_data_tensor = torch.tensor(batch_data, dtype=torch.float)
            real = (train_data_tensor - train_data_tensor.mean()) / train_data_tensor.std()
            valid = torch.ones((train_data_tensor.shape[0], 1))
            fake = torch.zeros((train_data_tensor.shape[0], 1))

            optimizer_G.zero_grad()
            encoded = enc(real)
            decoded = dec(encoded)
            g_loss = 0.01 * adversarial_loss(disc(encoded), valid) + 0.99 * recon_loss(decoded, real)

            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            
            z = torch.normal(0, 1, (batch_data.shape[0], 32))

            # real and fake loss should be close
            # discriminator(z) should be close to 0
            real_loss = adversarial_loss(disc(z), valid)
            fake_loss = adversarial_loss(disc(encoded.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()
        

        # Validation of the model.
        for fold, (_, val_index) in enumerate(kf.split(test)):
            df_val = test[val_index]
            fold_recon_loss = []
            fold_adversarial_loss = []
            enc.eval()
            dec.eval()
            disc.eval()
            n_batch = len(df_val) // 32
            for i in range(n_batch):
                str_idx = i * 32
                end_idx = str_idx + 32
                with torch.no_grad():
                    val_tensor = torch.tensor(df_val[str_idx:end_idx], dtype=torch.float)
                    val_real = (val_tensor - val_tensor.mean()) / val_tensor.std()
                    val_encoded = enc(val_real)
                    val_decoded = dec(val_encoded)
                recon_loss_val = recon_loss(val_decoded, val_real)
                valid_val = torch.ones((val_real.shape[0], 1))
                adv_loss_val = adversarial_loss(disc(val_encoded), valid_val)
                fold_recon_loss.append(recon_loss_val.item())
                fold_adversarial_loss.append(adv_loss_val.item())
            recon_losses.append(np.mean(fold_recon_loss))
            adversarial_losses.append(np.mean(fold_adversarial_loss))
    avg_recon_loss = np.mean(recon_losses)
    avg_adversarial_loss = np.mean(adversarial_losses)

    return avg_recon_loss, avg_adversarial_loss


from optuna.trial import TrialState
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







