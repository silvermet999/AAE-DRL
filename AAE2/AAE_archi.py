"""-----------------------------------------------import libraries-----------------------------------------------"""
import os
import json
import torch
from torch.nn import BatchNorm1d, Linear, Module, Sequential, Tanh, Sigmoid, Dropout, LeakyReLU
from torch import cuda, exp, normal
from torchsummary import summary


"""-----------------------------------initialize variables for inputs and outputs-----------------------------------"""
cuda = True if cuda.is_available() else False
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
in_out = 119 # in for the enc/gen out for the dec
# out_in_dim = 100 # in for the dec and disc out for the enc/gen
z_dim = 32





"""---------------------------------------------backprop and hidden layers-------------------------------------------"""
def reparameterization(mu, logvar, z_dim):
    std = exp(logvar / 2)
    sampled_z = normal(0, 1, (mu.size(0), z_dim)).to("cuda")
    z = sampled_z * std + mu
    return z



"""----------------------------------------------------AAE blocks----------------------------------------------------"""
params = {
    "n_layers_e": 44,
    "n_layers_de": 37,
    "n_layers_di": 11,
    "n_units_e_l0": 369,
    "dropout_l0": 0.2912014074950425,
    "n_units_e_l1": 264,
    "dropout_l1": 0.35957829905370775,
    "n_units_e_l2": 143,
    "dropout_l2": 0.16634929998161474,
    "n_units_e_l3": 418,
    "dropout_l3": 0.14488264346896107,
    "n_units_e_l4": 483,
    "dropout_l4": 0.34961088800051693,
    "n_units_e_l5": 340,
    "dropout_l5": 0.2882157846404448,
    "n_units_e_l6": 374,
    "dropout_l6": 0.02374574253753159,
    "n_units_e_l7": 185,
    "dropout_l7": 0.2984299208956127,
    "n_units_e_l8": 152,
    "dropout_l8": 0.07317829255139408,
    "n_units_e_l9": 225,
    "dropout_l9": 0.4633390011826975,
    "n_units_e_l10": 248,
    "dropout_l10": 0.18925529370010746,
    "n_units_e_l11": 406,
    "dropout_l11": 0.04124974957292793,
    "n_units_e_l12": 501,
    "dropout_l12": 0.3050765260201068,
    "n_units_e_l13": 252,
    "dropout_l13": 0.4529985369856203,
    "n_units_e_l14": 392,
    "dropout_l14": 0.09256666951563924,
    "n_units_e_l15": 313,
    "dropout_l15": 0.0034110266440671166,
    "n_units_e_l16": 166,
    "dropout_l16": 0.2433992663000255,
    "n_units_e_l17": 54,
    "dropout_l17": 0.3637684362506816,
    "n_units_e_l18": 142,
    "dropout_l18": 0.11901943868375703,
    "n_units_e_l19": 224,
    "dropout_l19": 0.12485899981978027,
    "n_units_e_l20": 321,
    "dropout_l20": 0.40862458111926414,
    "n_units_e_l21": 493,
    "dropout_l21": 0.25096111389024267,
    "n_units_e_l22": 303,
    "dropout_l22": 0.17344412654847136,
    "n_units_e_l23": 153,
    "dropout_l23": 0.19765134545072033,
    "n_units_e_l24": 138,
    "dropout_l24": 0.0293149257260214,
    "n_units_e_l25": 252,
    "dropout_l25": 0.43794389274878254,
    "n_units_e_l26": 110,
    "dropout_l26": 0.4365971670818343,
    "n_units_e_l27": 376,
    "dropout_l27": 0.4932216553741478,
    "n_units_e_l28": 245,
    "dropout_l28": 0.17462278881475202,
    "n_units_e_l29": 349,
    "dropout_l29": 0.04790680468013431,
    "n_units_e_l30": 478,
    "dropout_l30": 0.189794748643973,
    "n_units_e_l31": 424,
    "dropout_l31": 0.28708515210653035,
    "n_units_e_l32": 63,
    "dropout_l32": 0.12792539876826858,
    "n_units_e_l33": 204,
    "dropout_l33": 0.04126183161713565,
    "n_units_e_l34": 438,
    "dropout_l34": 0.46693733936443116,
    "n_units_e_l35": 390,
    "dropout_l35": 0.16244944823482776,
    "n_units_e_l36": 456,
    "dropout_l36": 0.3438666329016294,
    "n_units_e_l37": 294,
    "dropout_l37": 0.3571553789563026,
    "n_units_e_l38": 348,
    "dropout_l38": 0.30372651918722593,
    "n_units_e_l39": 239,
    "dropout_l39": 0.3367211118052325,
    "n_units_e_l40": 479,
    "dropout_l40": 0.44870344468741424,
    "n_units_e_l41": 260,
    "dropout_l41": 0.00375152328986611,
    "n_units_e_l42": 457,
    "dropout_l42": 0.34571830645315565,
    "n_units_e_l43": 310,
    "dropout_l43": 0.3001775340913507,
    "n_units_de_l0": 225,
    "n_units_de_l1": 281,
    "n_units_de_l2": 303,
    "n_units_de_l3": 444,
    "n_units_de_l4": 59,
    "n_units_de_l5": 124,
    "n_units_de_l6": 245,
    "n_units_de_l7": 293,
    "n_units_de_l8": 442,
    "n_units_de_l9": 444,
    "n_units_de_l10": 494,
    "n_units_de_l11": 438,
    "n_units_de_l12": 414,
    "n_units_de_l13": 83,
    "n_units_de_l14": 255,
    "n_units_de_l15": 420,
    "n_units_de_l16": 315,
    "n_units_de_l17": 473,
    "n_units_de_l18": 52,
    "n_units_de_l19": 161,
    "n_units_de_l20": 501,
    "n_units_de_l21": 88,
    "n_units_de_l22": 439,
    "n_units_de_l23": 441,
    "n_units_de_l24": 257,
    "n_units_de_l25": 313,
    "n_units_de_l26": 472,
    "n_units_de_l27": 326,
    "n_units_de_l28": 382,
    "n_units_de_l29": 187,
    "n_units_de_l30": 188,
    "n_units_de_l31": 248,
    "n_units_de_l32": 98,
    "n_units_de_l33": 158,
    "n_units_de_l34": 35,
    "n_units_de_l35": 236,
    "n_units_de_l36": 233,
    "n_units_di_l0": 493,
    "n_units_di_l1": 445,
    "n_units_di_l2": 64,
    "n_units_di_l3": 290,
    "n_units_di_l4": 359,
    "n_units_di_l5": 421,
    "n_units_di_l6": 403,
    "n_units_di_l7": 118,
    "n_units_di_l8": 162,
    "n_units_di_l9": 134,
    "n_units_di_l10": 466,

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
            dropout = params[f'dropout_l{i}']
            seq.append(Dropout(dropout))

        self.seq = Sequential(*seq)
        self.mu = Linear(310, 32)
        self.logvar = Linear(310, 32)


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
        seq += [
            Linear(out_features, in_out),
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
            seq.append(LeakyReLU())
            seq.append(BatchNorm1d(out_features))

        seq.append(Linear(out_features, 1))
        seq.append(Sigmoid())
        self.seq = Sequential(*seq)


    def forward(self, input_):
        return self.seq(input_)


encoder_generator = EncoderGenerator().cuda() if cuda else EncoderGenerator()
summary(encoder_generator, input_size=(in_out,))
decoder = Decoder().cuda() if cuda else Decoder()
summary(decoder, input_size=(z_dim,))
discriminator = Discriminator().cuda() if cuda else Discriminator()
summary(discriminator, input_size=(z_dim,))

with open('/home/silver/PycharmProjects/AAEDRL/AAE/hyperparams_g.json', 'r') as f:
    hyperparams_g = json.load(f)

with open('/home/silver/PycharmProjects/AAEDRL/AAE/hyperparams_d.json', 'r') as f:
    hyperparams_d = json.load(f)
