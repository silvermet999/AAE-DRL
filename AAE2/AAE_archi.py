"""-----------------------------------------------import libraries-----------------------------------------------"""
import os
from torch.nn import BatchNorm1d, Linear, Module, Sequential, Tanh, Sigmoid, Dropout, LeakyReLU
from torch import cuda, exp, normal


"""-----------------------------------initialize variables for inputs and outputs-----------------------------------"""
cuda = True if cuda.is_available() else False
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
in_out = 119 # in for the enc/gen out for the dec
# out_in_dim = 100 # in for the dec and disc out for the enc/gen
z_dim = 32
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"






"""---------------------------------------------backprop and hidden layers-------------------------------------------"""
def reparameterization(mu, logvar, z_dim):
    std = exp(logvar / 2)
    device = mu.device
    sampled_z = normal(0, 1, (mu.size(0), z_dim)).to(device)
    z = sampled_z * std + mu
    return z



"""----------------------------------------------------AAE blocks----------------------------------------------------"""
params = {
    'n_layers_e': 18, 'n_layers_de': 23, 'n_layers_di': 25, 'n_units_e_l0': 328, 'dropout_l0': 0.4508145404589306,
    'n_units_e_l1': 132, 'dropout_l1': 0.462657336072197, 'n_units_e_l2': 36, 'dropout_l2': 0.4811891655137515,
    'n_units_e_l3': 351, 'dropout_l3': 0.1325958233139718, 'n_units_e_l4': 83, 'dropout_l4': 0.2890545145126751,
    'n_units_e_l5': 325, 'dropout_l5': 0.31814229345131784, 'n_units_e_l6': 95, 'dropout_l6': 0.35264665462044575,
    'n_units_e_l7': 375, 'dropout_l7': 0.42977846340141135, 'n_units_e_l8': 124, 'dropout_l8': 0.13964236762852644,
    'n_units_e_l9': 467, 'dropout_l9': 0.4252003827871995, 'n_units_e_l10': 274, 'dropout_l10': 0.37891589475027376,
    'n_units_e_l11': 142, 'dropout_l11': 0.3732006358029489, 'n_units_e_l12': 434, 'dropout_l12': 0.4274768055805755,
    'n_units_e_l13': 36, 'dropout_l13': 0.35029290179513484, 'n_units_e_l14': 477, 'dropout_l14': 0.1279745833821307,
    'n_units_e_l15': 158, 'dropout_l15': 0.4378844177229787, 'n_units_e_l16': 270, 'dropout_l16': 0.4107626190212738,
    'n_units_e_l17': 379, 'dropout_l17': 0.36862218469927654, 'n_units_de_l0': 196, 'n_units_de_l1': 259,
    'n_units_de_l2': 53, 'n_units_de_l3': 35, 'n_units_de_l4': 256, 'n_units_de_l5': 113, 'n_units_de_l6': 440,
    'n_units_de_l7': 328, 'n_units_de_l8': 457, 'n_units_de_l9': 428, 'n_units_de_l10': 200, 'n_units_de_l11': 86,
    'n_units_de_l12': 138, 'n_units_de_l13': 317, 'n_units_de_l14': 183, 'n_units_de_l15': 128, 'n_units_de_l16': 320,
    'n_units_de_l17': 278, 'n_units_de_l18': 465, 'dropout_l18': 0.12847305916739454, 'n_units_de_l19': 46,
    'dropout_l19': 0.3909283296720884, 'n_units_de_l20': 491, 'dropout_l20': 0.34369909780464275, 'n_units_de_l21': 153,
    'dropout_l21': 0.05718230695211901, 'n_units_de_l22': 450, 'dropout_l22': 0.3130787165053842, 'n_units_di_l0': 156,
    'n_units_di_l1': 258, 'n_units_di_l2': 404, 'n_units_di_l3': 176, 'n_units_di_l4': 418, 'n_units_di_l5': 360,
    'n_units_di_l6': 115, 'n_units_di_l7': 317, 'n_units_di_l8': 37, 'n_units_di_l9': 375, 'n_units_di_l10': 331,
    'n_units_di_l11': 255, 'n_units_di_l12': 125, 'n_units_di_l13': 95, 'n_units_di_l14': 82, 'n_units_di_l15': 205,
    'n_units_di_l16': 111, 'n_units_di_l17': 462, 'n_units_di_l18': 366, 'n_units_di_l19': 43, 'n_units_di_l20': 189,
    'n_units_di_l21': 318, 'n_units_di_l22': 203, 'n_units_di_l23': 169, 'dropout_l23': 0.1396154156601358,
    'n_units_di_l24': 76, 'dropout_l24': 0.3336433169832094

}
class EncoderGenerator(Module):
    def __init__(self):
        super(EncoderGenerator, self).__init__()
        seq = []
        for i in range(params['n_layers_e']):
            in_features = in_out if i == 0 else params[f'n_units_e_l{i - 1}']
            out_features = params[f'n_units_e_l{i}']
            seq.append(Linear(in_features, out_features))
            seq.append(LeakyReLU())
            seq.append(BatchNorm1d(out_features))
            if i <= 17:
                dropout = params[f'dropout_l{i}']
                seq.append(Dropout(dropout))

        self.seq = Sequential(*seq)
        self.mu = Linear(379, 32)
        self.logvar = Linear(379, 32)


    def forward(self, x):
        x = self.seq(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar, z_dim)
        return mu, logvar, z


class Decoder(Module):
    def __init__(self):
        super(Decoder, self).__init__()
        seq = []
        for i in range(params['n_layers_de']):
            in_features = z_dim if i == 0 else params[f'n_units_de_l{i - 1}']
            out_features = params[f'n_units_de_l{i}']
            seq.append(Linear(in_features, out_features))
            seq.append(LeakyReLU())
            seq.append(BatchNorm1d(out_features))
            if 18 <= i <= 22:
                dropout = params[f'dropout_l{i}']
                seq.append(Dropout(dropout))
        seq.append(Linear(out_features, in_out))
        seq.append(Tanh())
        self.seq = Sequential(*seq)

    def forward(self, z):
        input_z = self.seq(z)
        return input_z




class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        seq = []
        for i in range(params['n_layers_di']):
            in_features = z_dim if i == 0 else params[f'n_units_di_l{i - 1}']
            out_features = params[f'n_units_di_l{i}']
            seq.append(Linear(in_features, out_features))
            seq.append(LeakyReLU())
            seq.append(BatchNorm1d(out_features))
            if i <= 23:
                dropout = params[f'dropout_l{i}']
                seq.append(Dropout(dropout))

        seq.append(Linear(out_features, 1))
        seq.append(Sigmoid())
        self.seq = Sequential(*seq)


    def forward(self, input_z):
        return self.seq(input_z)


encoder_generator = EncoderGenerator()
decoder = Decoder()
discriminator = Discriminator()


# with open('/home/silver/PycharmProjects/AAEDRL/AAE/hyperparams_g.json', 'r') as f:
#     hyperparams_g = json.load(f)
#
# with open('/home/silver/PycharmProjects/AAEDRL/AAE/hyperparams_d.json', 'r') as f:
#     hyperparams_d = json.load(f)
