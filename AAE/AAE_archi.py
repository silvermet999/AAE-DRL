"""-----------------------------------------------import libraries-----------------------------------------------"""
import os
import json
import torch
from torch.nn import BatchNorm1d, Linear, Module, Sequential, Tanh, Sigmoid, Dropout, LeakyReLU, Softmax
from torch import cuda, exp, normal
from torchsummary import summary


"""-----------------------------------initialize variables for inputs and outputs-----------------------------------"""
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
in_out = 105 # in for the enc/gen out for the dec
# out_in_dim = 100 # in for the dec and disc out for the enc/gen
z_dim = 32





"""---------------------------------------------backprop and hidden layers-------------------------------------------"""
def reparameterization(mu, logvar, z_dim):
    std = exp(logvar / 2)
    sampled_z = normal(0, 1, (mu.size(0), z_dim)).to("cuda:0")
    z = sampled_z * std + mu
    return z



"""----------------------------------------------------AAE blocks----------------------------------------------------"""
params = {
    'n_layers_e': 26, 'n_layers_de': 21, 'n_units_e_l0': 291, 'dropout_l0': 0.11869008882653087, 'n_units_e_l1': 350,
    'dropout_l1': 0.2244600208536538, 'n_units_e_l2': 211, 'dropout_l2': 0.3759205785067355, 'n_units_e_l3': 252,
    'dropout_l3': 0.36823113129866636, 'n_units_e_l4': 131, 'dropout_l4': 0.4976431121132785, 'n_units_e_l5': 200,
    'dropout_l5': 0.054043832415111404, 'n_units_e_l6': 125, 'dropout_l6': 0.32426135089738095, 'n_units_e_l7': 482,
    'dropout_l7': 0.004072887711605577, 'n_units_e_l8': 358, 'dropout_l8': 0.1275449214975269, 'n_units_e_l9': 497,
    'dropout_l9': 0.3895469060240036, 'n_units_e_l10': 507, 'dropout_l10': 0.4000784132471235, 'n_units_e_l11': 129,
    'dropout_l11': 0.15450535177563135, 'n_units_e_l12': 362, 'dropout_l12': 0.1879448807460054, 'n_units_e_l13': 391,
    'dropout_l13': 0.31660484775080355, 'n_units_e_l14': 452, 'dropout_l14': 0.48504643080261145, 'n_units_e_l15': 320,
    'dropout_l15': 0.2700967686150639, 'n_units_e_l16': 36, 'dropout_l16': 0.24205793583080354, 'n_units_e_l17': 157,
    'dropout_l17': 0.16317148147048893, 'n_units_e_l18': 293, 'dropout_l18': 0.25143542783073497, 'n_units_e_l19': 235,
    'dropout_l19': 0.10542930792051308, 'n_units_e_l20': 290, 'dropout_l20': 0.4556656361493996, 'n_units_e_l21': 200,
    'dropout_l21': 0.2332258974758853, 'n_units_e_l22': 75, 'dropout_l22': 0.21708148449977654, 'n_units_e_l23': 187,
    'dropout_l23': 0.3225685184911873, 'n_units_e_l24': 172, 'dropout_l24': 0.26451380659772095, 'n_units_e_l25': 338,
    'dropout_l25': 0.015300282621798555, 'n_units_de_l0': 394, 'n_units_de_l1': 240, 'n_units_de_l2': 232,
    'n_units_de_l3': 272, 'n_units_de_l4': 132, 'n_units_de_l5': 230, 'n_units_de_l6': 404, 'n_units_de_l7': 289,
    'n_units_de_l8': 266, 'n_units_de_l9': 294, 'n_units_de_l10': 359, 'n_units_de_l11': 271, 'n_units_de_l12': 240,
    'n_units_de_l13': 358, 'n_units_de_l14': 317, 'n_units_de_l15': 461, 'n_units_de_l16': 409, 'n_units_de_l17': 411,
    'n_units_de_l18': 72, 'n_units_de_l19': 280, 'n_units_de_l20': 279, 'n_layers_di': 18, 'n_units_di_l0': 37,
    'dropout_l26': 0.3953108842666025, 'n_units_di_l1': 446, 'dropout_l27': 0.46275164457331813, 'n_units_di_l2': 298,
    'dropout_l28': 0.41439725774646496, 'n_units_di_l3': 336, 'dropout_l29': 0.38235687521714845, 'n_units_di_l4': 395,
    'dropout_l30': 0.4162961310002777, 'n_units_di_l5': 263, 'dropout_l31': 0.40417908930241075, 'n_units_di_l6': 197,
    'dropout_l32': 0.3063286162663691, 'n_units_di_l7': 302, 'dropout_l33': 0.45484754782914616, 'n_units_di_l8': 497,
    'dropout_l34': 0.23592196910697666, 'n_units_di_l9': 278, 'dropout_l35': 0.33283213596440575, 'n_units_di_l10': 320,
    'dropout_l36': 0.38006464784065536, 'n_units_di_l11': 502, 'dropout_l37': 0.15037293254215617, 'n_units_di_l12': 81,
    'dropout_l38': 0.4906454386348957, 'n_units_di_l13': 407, 'dropout_l39': 0.4439607716761734, 'n_units_di_l14': 302,
    'dropout_l40': 0.18318849312157814, 'n_units_di_l15': 322, 'dropout_l41': 0.345179737062177, 'n_units_di_l16': 286,
    'dropout_l42': 0.16306932470098257, 'n_units_di_l17': 65, 'dropout_l43': 0.4407439993652657

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
            if i <= 25:
                dropout = params[f'dropout_l{i}']
                seq.append(Dropout(dropout))

        self.seq = Sequential(*seq)
        self.mu = Linear(338, 32)
        self.logvar = Linear(338, 32)


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
            if i > 25:
                dropout = params[f'dropout_l{i}']
                seq.append(Dropout(dropout))

        seq.append(Linear(out_features, 1))
        seq.append(Sigmoid())
        self.seq = Sequential(*seq)


    def forward(self, input_):
        return self.seq(input_)


encoder_generator = EncoderGenerator().to("cuda:0")
decoder = Decoder().to("cuda:1")
discriminator = Discriminator().to("cuda:1")

# with open('/home/silver/PycharmProjects/AAEDRL/AAE/hyperparams_g.json', 'r') as f:
#     hyperparams_g = json.load(f)
#
# with open('/home/silver/PycharmProjects/AAEDRL/AAE/hyperparams_d.json', 'r') as f:
#     hyperparams_d = json.load(f)


#  {'n_layers_di': 18, 'n_units_di_l0': 37, 'dropout_l0': 0.3953108842666025, 'n_units_di_l1': 446, 'dropout_l1': 0.46275164457331813, 'n_units_di_l2': 298, 'dropout_l2': 0.41439725774646496, 'n_units_di_l3': 336, 'dropout_l3': 0.38235687521714845, 'n_units_di_l4': 395, 'dropout_l4': 0.4162961310002777, 'n_units_di_l5': 263, 'dropout_l5': 0.40417908930241075, 'n_units_di_l6': 197, 'dropout_l6': 0.3063286162663691, 'n_units_di_l7': 302, 'dropout_l7': 0.45484754782914616, 'n_units_di_l8': 497, 'dropout_l8': 0.23592196910697666, 'n_units_di_l9': 278, 'dropout_l9': 0.33283213596440575, 'n_units_di_l10': 320, 'dropout_l10': 0.38006464784065536, 'n_units_di_l11': 502, 'dropout_l11': 0.15037293254215617, 'n_units_di_l12': 81, 'dropout_l12': 0.4906454386348957, 'n_units_di_l13': 407, 'dropout_l13': 0.4439607716761734, 'n_units_di_l14': 302, 'dropout_l14': 0.18318849312157814, 'n_units_di_l15': 322, 'dropout_l15': 0.345179737062177, 'n_units_di_l16': 286, 'dropout_l16': 0.16306932470098257, 'n_units_di_l17': 65, 'dropout_l17': 0.4407439993652657, 'lrd': 0.0003537881190733828, 'beta1d': 0.7986571988762406, 'beta2d': 0.9520849581646436}. Best is trial 6 with value: 0.6930056875944137.
