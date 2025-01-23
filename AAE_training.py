"""-----------------------------------------------import libraries-----------------------------------------------"""
import os
import pandas as pd
from sklearn.model_selection import KFold
import utils
from AAE import AAE_archi_opt
import torch
from torch.nn.functional import binary_cross_entropy, one_hot
import itertools

from data import main_u

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

cuda = True if torch.cuda.is_available() else False
torch.cuda.empty_cache()
torch.manual_seed(0)

"""--------------------------------------------------dataset and models--------------------------------------------------"""
# dataset = CustomDataset(main_u.X_train_sc.to_numpy(), main_u.y_train.to_numpy())
df = pd.DataFrame(pd.read_csv("/home/silver/PycharmProjects/AAEDRL/AAE/ds.csv"))[:141649]
X_train, X_test, y_train, y_test = main_u.vertical_split(df, main_u.y)
dataset_synth = utils.CustomDataset(X_train.to_numpy(), y_train.to_numpy())

in_out = 30
z_dim = 10
label_dim = 4

encoder_generator = AAE_archi_opt.EncoderGenerator(in_out, z_dim).cuda() if cuda else (
    AAE_archi_opt.EncoderGenerator(in_out, z_dim))

decoder = AAE_archi_opt.Decoder(z_dim+label_dim, in_out, utils.discrete, utils.continuous, utils.binary).cuda() if cuda \
    else (AAE_archi_opt.Decoder(z_dim+label_dim, in_out, utils.discrete, utils.continuous, utils.binary))

discriminator = AAE_archi_opt.Discriminator(z_dim, ).cuda() if cuda else (
    AAE_archi_opt.Discriminator(z_dim, ))


"""-----------------------------------------------------data gen-----------------------------------------------------"""
def save_features_to_csv(discrete_samples, continuous_samples, binary_samples):
    def dict_to_df(tensor_dict):
        all_data = []
        for sample_idx in range(next(iter(tensor_dict.values())).shape[0]):
            row_data = {}
            for feature_name, tensor in tensor_dict.items():
                if len(tensor.shape) > 2:
                    tensor = tensor.reshape(tensor.shape[0], -1)

                values = tensor[sample_idx].detach().cpu().numpy()
                if len(values.shape) == 0:
                    row_data[f"{feature_name}"] = values.item()
                else:
                    for _, value in enumerate(values):
                        row_data[f"{feature_name}"] = value
            all_data.append(row_data)
        return pd.DataFrame(all_data)

    discrete_df = dict_to_df(discrete_samples)
    continuous_df = dict_to_df(continuous_samples)
    binary_df = dict_to_df(binary_samples)

    combined_df = pd.concat([discrete_df, continuous_df, binary_df], axis=1)
    combined_df.to_csv('ds_synth.csv')

    return combined_df


def interpolate(z1, z2, n_steps=5):
    interpolations = []
    for alpha in torch.linspace(0, 1, n_steps):
        z = z1 * (1 - alpha) + z2 * alpha
        interpolations.append(z)
    return torch.stack(interpolations)

def sample_runs():
    discrete_samples = {feature: [] for feature in decoder.discrete_features}
    continuous_samples = {feature: [] for feature in decoder.continuous_features}
    binary_samples = {feature: [] for feature in decoder.binary_features}
    with torch.no_grad():
        n_interpolations = 4
        n_samples_per_interpolation = 35413
        z1 = torch.randn(n_interpolations, 14).cuda() if cuda else torch.randn(n_interpolations, 14)
        z2 = torch.randn(n_interpolations, 14).cuda() if cuda else torch.randn(n_interpolations, 14)

        for i in range(n_interpolations):
            interpolations = interpolate(z1[i], z2[i], n_samples_per_interpolation)
            discrete_out, continuous_out, binary_out = decoder(interpolations)

            discrete_samples, continuous_samples, binary_samples = utils.types_append(decoder,
                discrete_out, continuous_out, binary_out, discrete_samples, continuous_samples, binary_samples)

        discrete_samples, continuous_samples, binary_samples = utils.type_concat(decoder, discrete_samples,
                                                                                         continuous_samples, binary_samples)

        return discrete_samples, continuous_samples, binary_samples


"""-------------------------------------------------------KFold------------------------------------------------------"""

def get_kfold_indices(dataset, n_splits=3, shuffle=True, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    indices = [(train_indices, val_indices) for train_indices, val_indices in kf.split(dataset)]
    return indices

def kfold_cross_validation():
    kfold_indices = get_kfold_indices(utils.dataset_synth)
    fold_metrics = []
    for fold, indices in enumerate(kfold_indices, 1):
        train_loader, val_loader = utils.dataset_function(utils.dataset_synth, batch_size_t=32, batch_size_o=64,
                                                                  train=True)

        optimizer_G = torch.optim.SGD(itertools.chain(encoder_generator.parameters(), decoder.parameters()),
                                      lr=0.000001)
        optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=0.001)

        for epoch in range(33):
            g_loss, d_loss = train_model(train_loader, optimizer_G, optimizer_D)
            print(f"Epoch {epoch+1}/{33}, g loss: {g_loss}, d loss: {d_loss}")
            g_val, d_val = evaluate_model(val_loader)
        fold_metrics.append((g_val, d_val))
        print(f"Fold {fold} completed. Metrics: {fold_metrics[-1]}")

    torch.save(encoder_generator.state_dict(), "enc_gen_synth.pth")
    torch.save(decoder.state_dict(), "dec_synth.pth")
    torch.save(discriminator.state_dict(), "disc_synth.pth")

    d, c, b = sample_runs()
    save_features_to_csv(d, c, b)




"""--------------------------------------------------model training--------------------------------------------------"""
def train_model(train_loader, optimizer_G, optimizer_D):
    encoder_generator.train()
    decoder.train()
    discriminator.train()
    g_total = 0.0
    d_total = 0.0
    for i, (X, y) in enumerate(train_loader):
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

        optimizer_G.zero_grad()
        encoded = encoder_generator(real)
        dec_input = torch.cat([encoded, y], dim=1)
        discrete_outputs, continuous_outputs, binary_outputs = decoder(dec_input)

        g_loss = (0.1 * binary_cross_entropy(discriminator(encoded), valid) +
                  0.9 * decoder.compute_loss((discrete_outputs, continuous_outputs, binary_outputs),
                                               (discrete_targets, continuous_targets, binary_targets)))

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        z = utils.custom_dist((real.shape[0], z_dim)).cuda() if cuda else (
            utils.custom_dist((real.shape[0], z_dim)))
        real_loss = binary_cross_entropy(discriminator(z), valid)
        fake_loss = binary_cross_entropy(discriminator(encoded.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        g_total += g_loss.item()
        d_total += d_loss.item()

    g_total_loss = g_total / len(train_loader)
    d_total_loss = d_total / len(train_loader)

    return g_total_loss, d_total_loss


"""-------------------------------------------------model validation-------------------------------------------------"""

def evaluate_model(val_loader):
    encoder_generator.eval()
    decoder.eval()
    discriminator.eval()

    total_g_loss = 0.0
    total_d_loss = 0.0

    with torch.no_grad():
        for X, y in val_loader:
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

            g_loss = (0.1 * binary_cross_entropy(discriminator(encoded),
                                                 torch.ones((X.shape[0], 1),
                                                            requires_grad=False).cuda() if cuda else torch.ones(
                                                     (X.shape[0], 1), requires_grad=False)) +
                      0.9 * decoder.compute_loss((discrete_outputs, continuous_outputs, binary_outputs),
                                                 (discrete_targets, continuous_targets, binary_targets)))

            total_g_loss += g_loss.item()

            z = utils.custom_dist(
                (real.shape[0], z_dim)).cuda() if cuda else utils.custom_dist(
                (real.shape[0], z_dim))
            real_loss = binary_cross_entropy(discriminator(z), valid)
            fake_loss = binary_cross_entropy(discriminator(encoded.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

        avg_g_loss = total_g_loss / len(val_loader)
        avg_d_loss = total_d_loss / len(val_loader)

    return avg_g_loss, avg_d_loss




