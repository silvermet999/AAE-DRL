"""-----------------------------------------------import libraries-----------------------------------------------"""
import os
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import MultiStepLR
import AAE_archi

import numpy as np
import pandas as pd
import torch
from torch.nn import BCELoss, L1Loss
from torch import cuda
from torchsummary import summary
import mlflow
import main
import itertools

# import optim_hyperp

cuda = True if cuda.is_available() else False


"""--------------------------------------------------loss and optim--------------------------------------------------"""
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
# mlflow.set_experiment("MLflow Quickstart")

adversarial_loss = BCELoss().cuda() if cuda else BCELoss()
recon_loss = L1Loss().cuda() if cuda else L1Loss()

encoder_generator = AAE_archi.EncoderGenerator().cuda() if cuda else AAE_archi.EncoderGenerator()
summary(encoder_generator, input_size=(AAE_archi.in_out,))
decoder = AAE_archi.Decoder().cuda() if cuda else AAE_archi.Decoder()
summary(decoder, input_size=(AAE_archi.z_dim,))
discriminator = AAE_archi.Discriminator().cuda() if cuda else AAE_archi.Discriminator()
summary(discriminator, input_size=(AAE_archi.z_dim,))


# optimizer_G = torch.optim.Adam(
#     itertools.chain(encoder_generator.parameters(), decoder.parameters()), lr=optim_hyperp.best["lr"], betas=(optim_hyperp.best["beta1"], optim_hyperp.best["beta2"]), weight_decay=optim_hyperp.best["weight_decay"])
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=optim_hyperp.best["lr"], betas=(optim_hyperp.best["beta1"], optim_hyperp.best["beta2"]), weight_decay=optim_hyperp.best["weight_decay"])
# scheduler_D = MultiStepLR(optimizer_D, milestones=[optim_hyperp.best["low"], optim_hyperp.best["high"]], gamma=optim_hyperp.best["gamma"])
# scheduler_G = MultiStepLR(optimizer_G, milestones=[optim_hyperp.best["low"], optim_hyperp.best["high"]], gamma=optim_hyperp.best["gamma"])


optimizer_G = torch.optim.Adam(
    itertools.chain(encoder_generator.parameters(), decoder.parameters()), lr=.00920828229062108, betas=(.9758534841202955, .9918395879014463), weight_decay=.0008904987305687305)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1.0846450506796643e-05, betas=(.8943741362785895, .9995314706647865), weight_decay=.0004259377371418141)
scheduler_D = MultiStepLR(optimizer_D, milestones=[40, 89], gamma=.017159877576271722)
scheduler_G = MultiStepLR(optimizer_G, milestones=[28, 89], gamma=.09240555734722164)



Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



"""-----------------------------------------------------data gen-----------------------------------------------------"""
def get_next_file_number():
    counter_file = "runs/file_counter.txt"
    if not os.path.exists(counter_file):
        with open(counter_file, "w") as f:
            f.write("0")
        return 0
    with open(counter_file, "r") as f:
        counter = int(f.read())
    with open(counter_file, "w") as f:
        f.write(str(counter + 1))
    return counter
file_number = get_next_file_number()


def sample_runs(n_row, z_dim):
    log_normal = torch.distributions.LogNormal(loc=0, scale=1)
    z = log_normal.sample((n_row ** 2, z_dim)).to(device="cuda") if cuda else log_normal.sample((n_row ** 2, z_dim))
    gen_input = decoder(z).cpu()
    gen_data = gen_input.data.numpy()

    gen_df = pd.DataFrame(gen_data, columns=main.X.columns)
    filename = f"runs/uns{file_number}.csv"
    gen_df.to_csv(filename, index=False)


"""--------------------------------------------------model training--------------------------------------------------"""
for epoch in range(100):
    n_batch = len(main.X_train_rs) // 16
    for i in range(n_batch):
        str_idx = i * 16
        end_idx = str_idx + 16
        batch_data = main.X_train_rs[str_idx:end_idx]
        train_data_tensor = torch.tensor(batch_data, dtype=torch.float).cuda() if cuda else torch.tensor(batch_data, dtype=torch.float)

        real = (train_data_tensor - train_data_tensor.mean()) / train_data_tensor.std()
        valid = torch.ones((train_data_tensor.shape[0], 1)).cuda() if cuda else torch.ones((train_data_tensor.shape[0], 1))
        fake = torch.zeros((train_data_tensor.shape[0], 1)).cuda() if cuda else torch.zeros((train_data_tensor.shape[0], 1))

        optimizer_G.zero_grad()
        encoded = encoder_generator(real)
        decoded = decoder(encoded)
        g_loss = 0.01 * adversarial_loss(discriminator(encoded), valid) + 0.99 * recon_loss(decoded, real)

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        log_normal = torch.distributions.LogNormal(loc=0, scale=1)
        z = log_normal.sample((batch_data.shape[0], AAE_archi.z_dim)).to(cuda) if cuda else log_normal.sample((batch_data.shape[0], AAE_archi.z_dim))

        # real and fake loss should be close
        # discriminator(z) should be close to 0
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

    scheduler_G.step()
    scheduler_D.step()
    print(epoch, d_loss.item(), g_loss.item())

    batches_done = epoch * len(main.X_train_rs)
    if batches_done % 400 == 0:
        sample_runs(n_row=179, z_dim=AAE_archi.z_dim)

    """-------------------------------------------------save model---------------------------------------------------"""
    if (epoch + 1) % 20 == 0:
        torch.save(encoder_generator.state_dict(), f'encoder_generator_epoch_{epoch + 1}.pth')
        torch.save(decoder.state_dict(), f'decoder_epoch_{epoch + 1}.pth')
        torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch + 1}.pth')

torch.save(encoder_generator.state_dict(), 'encoder_generator_final.pth')
torch.save(decoder.state_dict(), 'decoder_final.pth')
torch.save(discriminator.state_dict(), 'discriminator_final.pth')

encoder_generator.load_state_dict(torch.load('encoder_generator_final.pth'))
decoder.load_state_dict(torch.load('decoder_final.pth'))
discriminator.load_state_dict(torch.load('discriminator_final.pth'))



"""--------------------------------------------------model testing---------------------------------------------------"""
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
recon_losses = []
adversarial_losses = []
for fold, (_, val_index) in enumerate(kf.split(main.X_test_rs)):
    print(f"Fold {fold + 1}/{n_splits}")
    df_val = main.X_test_rs[val_index]
    fold_recon_loss = []
    fold_adversarial_loss = []
    encoder_generator.eval()
    decoder.eval()
    discriminator.eval()
    n_batch = len(df_val) // 1
    for i in range(n_batch):
        str_idx = i * 1
        end_idx = str_idx + 1
        with torch.no_grad():
            val_tensor = torch.tensor(df_val[str_idx:end_idx], dtype=torch.float).cuda() if cuda else torch.tensor(df_val[str_idx:end_idx], dtype=torch.float)
            val_real = (val_tensor - val_tensor.mean()) / val_tensor.std()
            val_encoded = encoder_generator(val_real)
            val_decoded = decoder(val_encoded)
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

with mlflow.start_run():
    mlflow.set_tag("Training Info", "Test")

    mlflow.log_metric("test_avg_recon_loss", avg_recon_loss)
    mlflow.log_metric("test_std_recon_loss", std_recon_loss)
    mlflow.log_metric("test_avg_adversarial_loss", avg_adversarial_loss)
    mlflow.log_metric("test_std_adversarial_loss", std_adversarial_loss)

    # Log individual fold results
    for fold, (recon_loss, adv_loss) in enumerate(zip(recon_losses, adversarial_losses)):
        mlflow.log_metric(f"test_fold_{fold + 1}_recon_loss", recon_loss)
        mlflow.log_metric(f"test_fold_{fold + 1}_adv_loss", adv_loss)

    mlflow.set_tag("Model Info", "Adversarial Autoencoder")
    mlflow.set_tag("Evaluation", "5-Fold Cross-Validation")

    model_info_gen = mlflow.sklearn.log_model(
        sk_model = encoder_generator,
        artifact_path="mlflow/gen",
        input_example=AAE_archi.in_out,
        registered_model_name="supervised_G_tracking",
    )
    model_info_disc = mlflow.sklearn.log_model(
        sk_model=discriminator,
        artifact_path="mlflow/discriminator",
        input_example=AAE_archi.z_dim,
        registered_model_name="supervisedD_tracking",
    )

