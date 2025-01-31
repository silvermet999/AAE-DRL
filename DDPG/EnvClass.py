# libraries
import torch
from torch.nn import Module, BCELoss, CrossEntropyLoss
import os

from torch.nn.functional import one_hot

import utils


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
cuda = True if torch.cuda.is_available() else False

class Env(Module):
    def __init__(self, model_G, model_D, model_de, classifier):
        super(Env, self).__init__()

        self._state = None
        self.decoder = model_de
        self.generator = model_G
        self.disciminator = model_D
        self.classifier = classifier

        self.d_reward_coeff = 0.3
        self.cl_reward_coeff = 0.1
        self.bin = BCELoss().cuda() if cuda else BCELoss()
        self.ce = CrossEntropyLoss().cuda() if cuda else CrossEntropyLoss()

        self.count = 0

    def reset(self):
        self.count = 0

    def set_state(self, state):
        self._state = state
        return state.detach().cpu().numpy()

    def forward(self, action, disc, episode_target):
        d_decoded = {feature: [] for feature in self.decoder.discrete_features}
        c_decoded = {feature: [] for feature in self.decoder.continuous_features}
        b_decoded = {feature: [] for feature in self.decoder.binary_features}
        with (torch.no_grad()):
            episode_target = episode_target.to(torch.long).cuda() if cuda else episode_target.to(torch.long)
            labels = one_hot(episode_target, num_classes=4)
            z_cont = torch.tensor(action).cuda() if cuda else torch.tensor(action)
            z_disc = torch.tensor(list(disc.values())).cuda() if cuda else torch.tensor(list(disc.values()))
            z_disc = z_disc.expand(z_cont.size(0), -1).cuda() if cuda else z_disc.expand(z_cont.size(0), -1)
            labels = labels.squeeze(1)
            z = torch.concat([z_cont, z_disc, labels], 1)
            d, c, b = self.decoder(z)
            d_decoded, c_decoded, b_decoded = utils.types_append(self.decoder, d, c, b, d_decoded, c_decoded, b_decoded)
            d_decoded, c_decoded, b_decoded = utils.type_concat(self.decoder, d_decoded, c_decoded, b_decoded)
            decoded = utils.all_samples(d_decoded, c_decoded, b_decoded)
            gen_out = self.generator(decoded)
            dis_judge = self.disciminator(gen_out)
            classifier_output, _ = self.classifier.encoder(decoded)
            classifier_logits, _ = self.classifier.classify(classifier_output)

        if len(episode_target.shape) == 2:
            episode_target = episode_target.squeeze(1)
        reward_cl = self.cl_reward_coeff * self.ce(classifier_logits, episode_target).cpu().data.numpy()
        reward_d = self.d_reward_coeff * self.bin(dis_judge, torch.ones_like(dis_judge)).cpu().data.numpy()

        reward = reward_cl + reward_d
        done = True
        self.count += 1

        next_state = decoded.detach().cpu().data.numpy()
        self._state = decoded
        return next_state, reward, done
