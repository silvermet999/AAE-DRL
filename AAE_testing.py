"""-----------------------------------------------import libraries-----------------------------------------------"""
import os
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import MultiStepLR

from AAE import AAE_archi

import numpy as np
import torch
from torch.nn import BCELoss, L1Loss
from torch import cuda
from data import main
import itertools
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"



cuda = True if cuda.is_available() else False
torch.cuda.empty_cache()


df_train = main.X_test_rs


"""--------------------------------------------------loss and optim--------------------------------------------------"""
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
# mlflow.set_experiment("MLflow Quickstart")

adversarial_loss = BCELoss().cuda() if cuda else BCELoss()
recon_loss = L1Loss().cuda() if cuda else L1Loss()


encoder_generator = AAE_archi.EncoderGenerator().cuda() if cuda else AAE_archi.EncoderGenerator()
decoder = AAE_archi.Decoder().cuda() if cuda else AAE_archi.Decoder()
discriminator = AAE_archi.Discriminator().cuda() if cuda else AAE_archi.Discriminator()



hyperparams_g = {'lr': 0.001, 'beta1': 0.5, 'beta2': 0.999}
hyperperams_d={'lrd':0.001, 'beta1d': 0.9, 'beta2d': 0.99} # stuck in local min


optimizer_G = torch.optim.Adam(
    itertools.chain(encoder_generator.parameters(), decoder.parameters()), lr=hyperparams_g["lr"], betas=(hyperparams_g["beta1"], hyperparams_g["beta2"]))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=hyperperams_d["lrd"], betas=(hyperperams_d["beta1d"], hyperperams_d["beta2d"]))
scheduler_D = MultiStepLR(optimizer_D, milestones=[30, 80], gamma=0.01)

scheduler_G = MultiStepLR(optimizer_G, milestones=[30, 80], gamma=0.01)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


"""--------------------------------------------------model training--------------------------------------------------"""
for epoch in range(100):
    n_batch = len(df_train) // 64
    for i in range(n_batch):
        str_idx = i * 64
        end_idx = str_idx+64
        train_data_tensor = torch.tensor(df_train[str_idx:end_idx], dtype=torch.float).cuda() if cuda else torch.tensor(df_train[str_idx:end_idx], dtype=torch.float)

        real = (train_data_tensor - train_data_tensor.mean()) / train_data_tensor.std()
        valid = torch.ones((train_data_tensor.shape[0], 1)).cuda() if cuda else torch.ones((train_data_tensor.shape[0], 1))
        fake = torch.zeros((train_data_tensor.shape[0], 1)).cuda() if cuda else torch.zeros((train_data_tensor.shape[0], 1))

        optimizer_G.zero_grad()
        encoded = encoder_generator(real)
        labels_tensor = AAE_archi.encoded_tensor[str_idx:end_idx]
        dec_input = torch.cat([encoded, labels_tensor], dim=1)
        decoded = decoder(dec_input)
        g_loss = 0.001 * adversarial_loss(discriminator(encoded), valid) + 0.999 * recon_loss(decoded, real)

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        # log_normal = torch.distributions.LogNormal(loc=0, scale=1)
        # z = log_normal.sample((batch_data.shape[0], AAE_archi.z_dim)).cuda() if cuda else log_normal.sample((batch_data.shape[0], AAE_archi.z_dim))
        z = torch.normal(0, 1, (train_data_tensor.shape[0], AAE_archi.z_dim)).cuda() if cuda else torch.normal(0, 1, (train_data_tensor.shape[0], AAE_archi.z_dim))

        # real and fake loss should be close
        # discriminator(z) should be close to 0e
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

    # scheduler_G.step()
    # scheduler_D.step()
    print(epoch, d_loss.item(), g_loss.item())

    """-------------------------------------------------save model---------------------------------------------------"""
#     if (epoch + 1) % 20 == 0:
#         torch.save(encoder_generator.state_dict(), f'encoder_generator_epoch_{epoch + 1}.pth')
#         torch.save(decoder.state_dict(), f'decoder_epoch_{epoch + 1}.pth')
#         torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch + 1}.pth')
#
# torch.save(encoder_generator.state_dict(), 'encoder_generator_final.pth')
# torch.save(decoder.state_dict(), 'decoder_final.pth')
# torch.save(discriminator.state_dict(), 'discriminator_final.pth')
#
# encoder_generator.load_state_dict(torch.load('encoder_generator_final.pth'))
# decoder.load_state_dict(torch.load('decoder_final.pth'))
# discriminator.load_state_dict(torch.load('discriminator_final.pth'))

avg_g_loss = np.mean(g_loss.item())
std_g_loss = np.std(g_loss.item())


"""--------------------------------------------------model testing---------------------------------------------------"""
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
recon_losses = []
adversarial_losses = []
for fold, (_, val_index) in enumerate(kf.split(df_train)):
    print(f"Fold {fold + 1}/{n_splits}")
    df_val = df_train[val_index]
    fold_recon_loss = []
    fold_adversarial_loss = []
    encoder_generator.eval()
    decoder.eval()
    discriminator.eval()
    n_batch = len(df_val)
    for i in range(n_batch):
        str_idx = i * 1
        end_idx = str_idx + 1
        with torch.no_grad():
            val_tensor = torch.tensor(df_val[str_idx:end_idx], dtype=torch.float).cuda() if cuda else torch.tensor(df_val[str_idx:end_idx], dtype=torch.float)
            val_real = (val_tensor - val_tensor.mean()) / val_tensor.std()
            val_encoded = encoder_generator(val_real)
            labels_tensor = AAE_archi.encoded_tensor[str_idx:end_idx]
            dec_input = torch.cat([val_encoded, labels_tensor], dim=1)
            val_decoded = decoder(dec_input)
        recon_loss_val = recon_loss(val_decoded, val_real)
        valid_val = torch.ones((val_real.shape[0], 1)).cuda() if cuda else torch.ones((val_real.shape[0], 1))
        adv_loss_val = adversarial_loss(discriminator(val_encoded), valid_val)
        fold_recon_loss.append(recon_loss_val.item())
        fold_adversarial_loss.append(adv_loss_val.item())
    recon_losses.append(np.mean(fold_recon_loss))
    adversarial_losses.append(np.mean(fold_adversarial_loss))
avg_recon_loss = np.mean(recon_losses)
avg_adversarial_loss = np.mean(adversarial_losses)
std_recon_loss = np.std(recon_losses)
std_adversarial_loss = np.std(adversarial_losses)
print(f"Average Reconstruction Loss: {avg_recon_loss:.4f} ± {std_recon_loss:.4f}")
print(f"Average Adversarial Loss: {avg_adversarial_loss:.4f} ± {std_adversarial_loss:.4f}")


"""--------------------------------------------------mlflow---------------------------------------------------"""

# with mlflow.start_run():
#     mlflow.set_tag("Training Info", "Test")
#
#     mlflow.log_metric("test_avg_recon_loss", avg_recon_loss)
#     mlflow.log_metric("test_std_recon_loss", std_recon_loss)
#     mlflow.log_metric("test_avg_adversarial_loss", avg_adversarial_loss)
#     mlflow.log_metric("test_std_adversarial_loss", std_adversarial_loss)
#
#     # Log individual fold results
#     for fold, (recon_loss, adv_loss) in enumerate(zip(recon_losses, adversarial_losses)):
#         mlflow.log_metric(f"test_fold_{fold + 1}_recon_loss", recon_loss)
#         mlflow.log_metric(f"test_fold_{fold + 1}_adv_loss", adv_loss)
#
#     mlflow.set_tag("Model Info", "Adversarial Autoencoder")
#     mlflow.set_tag("Evaluation", "5-Fold Cross-Validation")
#
#     model_info_gen = mlflow.sklearn.log_model(
#         sk_model = encoder_generator,
#         artifact_path="mlflow/gen",
#         input_example=AAE_archi.in_out,
#         registered_model_name="supervised_G_tracking",
#     )
#     model_info_disc = mlflow.sklearn.log_model(
#         sk_model=discriminator,
#         artifact_path="mlflow/discriminator",
#         input_example=AAE_archi.z_dim,
#         registered_model_name="supervisedD_tracking",
#     )
# Average Reconstruction Loss: 0.4099 ± 0.0106
# Average Adversarial Loss: 0.7027 ± 0.0202

