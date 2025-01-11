# libraries
import torch
import torch.nn as nn
import os
import utils
import numpy as np

from AAE import AAE_archi_opt

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
cuda = True if torch.cuda.is_available() else False

class Env(nn.Module):
    def __init__(self, model_G, model_D, model_de, classifier):
        super(Env, self).__init__()

        self._state = None
        self.decoder = model_de
        self.generator = model_G
        self.disciminator = model_D
        self.classifier = classifier

        self.d_reward_coeff = 1
        self.cl_reward_coeff = 0.5
        self.bin = torch.nn.BCELoss().cuda() if cuda else torch.nn.BCELoss()
        self.ce = torch.nn.CrossEntropyLoss().cuda() if cuda else torch.nn.CrossEntropyLoss()

        self.count = 0

    def reset(self):
        self.count = 0

    def set_state(self, state):
        self._state = state
        return state.detach().cpu().numpy()

    def forward(self, action, disc, episode_target, t=None):
        d_decoded = {feature: [] for feature in self.decoder.discrete_features}
        c_decoded = {feature: [] for feature in self.decoder.continuous_features}
        # b_decoded = {feature: [] for feature in self.decoder.binary_features}
        with (torch.no_grad()):
            z_cont = torch.tensor(action).cuda() if cuda else torch.tensor(action)
            z_disc = torch.tensor(list(disc.values())).cuda() if cuda else torch.tensor(list(disc.values()))
            z_disc = z_disc.unsqueeze(0).expand(z_cont.size(0), -1)
            z = torch.concat([z_disc, z_cont], 1)
            d, c = self.decoder.disc_cont(z)
            d_decoded, c_decoded = AAE_archi_opt.types_append(d, c, d_decoded, c_decoded)
            d_decoded, c_decoded = AAE_archi_opt.type_concat(d_decoded, c_decoded)
            decoded = utils.all_samples(d_decoded, c_decoded)
            print("decoded", decoded.shape)
            gen_out = self.generator(decoded)
            dis_judge = self.disciminator(gen_out)
            classifier_output, _ = self.classifier.encoder(decoded)
            classifier_logits, _ = self.classifier.classify(classifier_output)

        episode_target = episode_target.to(torch.long).cuda() if cuda else episode_target.to(torch.long)
        reward_cl = self.cl_reward_coeff * abs(self.ce(classifier_logits, episode_target).cpu().data.numpy())
        reward_d = self.d_reward_coeff * self.bin(dis_judge, torch.ones_like(dis_judge)).cpu().data.numpy()

        reward = reward_cl + reward_d

        done = True


        self.count += 1

        # the nextState
        next_state = decoded.detach().cpu().data.numpy()
        self._state = decoded
        return next_state, reward, done, classifier_logits.cpu().data.numpy()
