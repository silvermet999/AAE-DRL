"""-----------------------------------------------import libraries-----------------------------------------------"""
import os

import torch
from torch.nn import BatchNorm1d, ReLU, Linear, Module, Sequential, Tanh, Sigmoid, Dropout, Conv1d
from torch import cuda, exp

"""-----------------------------------initialize variables for inputs and outputs-----------------------------------"""
# if unsupervised in_out is 105 else 106
# if clean and unsupervised  else
cuda = True if cuda.is_available() else False
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
in_out = 105 # in for the enc/gen out for the dec
# out_in_dim = 100 # in for the dec and disc out for the enc/gen
z_dim = 32





"""---------------------------------------------backprop and hidden layers-------------------------------------------"""
def reparameterization(mu, logvar, z_dim):
    std = exp(logvar / 2)
    device = mu.device
    sampled_z = torch.normal(0, 1, (mu.size(0), z_dim)).to(device)
    z = sampled_z * std + mu
    return z



"""----------------------------------------------------AAE blocks----------------------------------------------------"""
params = {
    "n_layers_e": 22,
    "n_layers_de": 39,
    "n_layers_di": 13,
    "n_units_e_l0": 308,
    "dropout_l0": 0.06008920296932768,
    "n_units_e_l1": 182,
    "dropout_l1": 0.023293631300132688,
    "n_units_e_l2": 342,
    "dropout_l2": 0.03232757632776423,
    "n_units_e_l3": 307,
    "dropout_l3": 0.15180095593606227,
    "n_units_e_l4": 245,
    "dropout_l4": 0.236119257149791,
    "n_units_e_l5": 77,
    "dropout_l5": 0.2304072245004382,
    "n_units_e_l6": 432,
    "dropout_l6": 0.01988985826280406,
    "n_units_e_l7": 107,
    "dropout_l7": 0.16729083469687492,
    "n_units_e_l8": 247,
    "dropout_l8": 0.06991869537030092,
    "n_units_e_l9": 142,
    "dropout_l9": 0.17667591332548163,
    "n_units_e_l10": 363,
    "dropout_l10": 0.04388109281921961,
    "n_units_e_l11": 410,
    "dropout_l11": 0.0363551898869714,
    "n_units_e_l12": 431,
    "dropout_l12": 0.2899213371874036,
    "n_units_e_l13": 420,
    "dropout_l13": 0.016597937298508558,
    "n_units_e_l14": 332,
    "dropout_l14": 0.16268368295161814,
    "n_units_e_l15": 241,
    "dropout_l15": 0.26260902972986416,
    "n_units_e_l16": 165,
    "dropout_l16": 0.2296256510944324,
    "n_units_e_l17": 509,
    "dropout_l17": 0.13182299873189707,
    "n_units_e_l18": 398,
    "dropout_l18": 0.11241958518461637,
    "n_units_e_l19": 465,
    "dropout_l19": 0.09815635605307504,
    "n_units_e_l20": 231,
    "dropout_l20": 0.13725165097950284,
    "n_units_e_l21": 61,
    "dropout_l21": 0.1327773037647823,
    "n_units_l1": 296,
    "dropout_l22": 0.24080702621841588,
    "n_units_l2": 168,
    "dropout_l23": 0.07713299220509373,
    "n_units_de_l0": 446,
    "n_units_de_l1": 471,
    "n_units_de_l2": 475,
    "n_units_de_l3": 507,
    "n_units_de_l4": 472,
    "n_units_de_l5": 114,
    "n_units_de_l6": 379,
    "n_units_de_l7": 319,
    "n_units_de_l8": 469,
    "n_units_de_l9": 470,
    "n_units_de_l10": 134,
    "n_units_de_l11": 222,
    "n_units_de_l12": 478,
    "n_units_de_l13": 172,
    "n_units_de_l14": 442,
    "n_units_de_l15": 110,
    "n_units_de_l16": 90,
    "n_units_de_l17": 256,
    "n_units_de_l18": 271,
    "n_units_de_l19": 93,
    "n_units_de_l20": 42,
    "n_units_de_l21": 474,
    "n_units_de_l22": 266,
    "n_units_de_l23": 359,
    "n_units_de_l24": 395,
    "dropout_l24": 0.15141577488125124,
    "n_units_de_l25": 138,
    "dropout_l25": 0.07967880619503673,
    "n_units_de_l26": 455,
    "dropout_l26": 0.11756302876363905,
    "n_units_de_l27": 381,
    "dropout_l27": 0.055008738720332,
    "n_units_de_l28": 457,
    "dropout_l28": 0.0023431468802443288,
    "n_units_de_l29": 224,
    "dropout_l29": 0.13699879333696777,
    "n_units_de_l30": 197,
    "dropout_l30": 0.15479281822463808,
    "n_units_de_l31": 114,
    "dropout_l31": 0.27142385012732256,
    "n_units_de_l32": 365,
    "dropout_l32": 0.08208512395623936,
    "n_units_de_l33": 166,
    "dropout_l33": 0.2535872073559903,
    "n_units_de_l34": 254,
    "dropout_l34": 0.1342735836046149,
    "n_units_de_l35": 478,
    "dropout_l35": 0.29004912046046777,
    "n_units_de_l36": 178,
    "dropout_l36": 0.22767910311832157,
    "n_units_de_l37": 217,
    "dropout_l37": 0.20997449586128783,
    "n_units_de_l38": 447,
    "dropout_l38": 0.04484944061327628,
    "n_units_t": 85,
    "dropout_l39": 0.15703283114499253,
    "n_units_di_l0": 35,
    "n_units_di_l1": 109,
    "n_units_di_l2": 126,
    "n_units_di_l3": 88,
    "n_units_di_l4": 53,
    "n_units_di_l5": 48,
    "n_units_di_l6": 97,
    "n_units_di_l7": 68,
    "n_units_di_l8": 81,
    "n_units_di_l9": 94,
    "n_units_di_l10": 47,
    "n_units_di_l11": 72,
    "n_units_di_l12": 48 }
class EncoderGenerator(Module):
    def __init__(self):
        super(EncoderGenerator, self).__init__()
        seq = []
        for i in range(params['n_layers_e']):
            in_features = in_out if i == 0 else params[f'n_units_e_l{i - 1}']
            out_features = params[f'n_units_e_l{i}']
            seq.append(Linear(in_features, out_features))
            seq.append(ReLU())
            seq.append(BatchNorm1d(out_features))
            if i <= 21:
                dropout = params[f'dropout_l{i}']
                seq.append(Dropout(dropout))

        seq += [
            Linear(61, 296),
            Dropout(.1327773037647823),
            Linear(296, 168),
            Dropout(0.24080702621841588),
            Linear(168, 32),
            Dropout(.07713299220509373)
        ]
        self.seq = Sequential(*seq)
        self.mu = Linear(32, 32)
        self.logvar = Linear(32, 32)


    def forward(self, x):
        # x_f = x_o.view(x_o.shape[0], -1)
        x = self.seq(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar, z_dim)
        return z


class Decoder(Module):
    def __init__(self):
        super(Decoder, self).__init__()
        seq = []
        for i in range(params['n_layers_de']):
            in_features = z_dim if i == 0 else params[f'n_units_de_l{i - 1}']
            out_features = params[f'n_units_de_l{i}']
            seq.append(Linear(in_features, out_features))
            seq.append(ReLU())
            seq.append(BatchNorm1d(out_features))
            if i > 23:
                dropout = params[f'dropout_l{i}']
                seq.append(Dropout(dropout))
        seq += [
            Linear(out_features, 85),
            Dropout(0.15703283114499253),
            Tanh()
        ]
        self.seq = Sequential(*seq)

    def forward(self, z):
        input_ = self.seq(z)
        return input_




class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        seq = []
        for i in range(params['n_layers_di']):
            in_features = z_dim if i == 0 else params[f'n_units_di_l{i - 1}']
            out_features = params[f'n_units_di_l{i}']
            seq.append(Linear(in_features, out_features))
            seq.append(ReLU())
            seq.append(BatchNorm1d(out_features))

        seq.append(Linear(out_features, 1))
        seq.append(Sigmoid)
        self.seq = Sequential(*seq)


    def forward(self, input_):
        return self.seq(input_)

from torchsummary import summary
encoder_generator = EncoderGenerator().cuda() if cuda else EncoderGenerator()
summary(encoder_generator, input_size=(in_out,))
decoder = Decoder().cuda() if cuda else Decoder()
summary(decoder, input_size=(z_dim,))
discriminator = Discriminator().cuda() if cuda else Discriminator()
summary(discriminator, input_size=(z_dim,))