# libraries
import torch
import torch.nn as nn
import os
import numpy as np
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

    def forward(self, action, episode_target, t=None):
        # episodeTarget: the number that the RL agent is trying to find
        with (torch.no_grad()):
            z = action.cuda() if cuda else action
            decoded = self.decoder(z)
            gen_out = self.generator(decoded)
            dis_judge = self.disciminator(gen_out)
            classifier_output, _ = self.classifier.encoder(decoded)
            classifier_logits, _ = self.classifier.classify(classifier_output)

        episode_target = episode_target.to(torch.float32).cuda() if cuda else episode_target.to(torch.float32)
        # reward_cl = self.cl_reward_coeff * (-self.ce(classifier_logits, episode_target).cpu().data.numpy())
        reward_cl = self.cl_reward_coeff * abs(self.ce(classifier_logits, episode_target).cpu().data.numpy())
        # reward_d = self.d_reward_coeff * (1 - (self.bin(dis_judge, torch.ones_like(dis_judge)).cpu().data.numpy()))
        reward_d = self.d_reward_coeff * self.bin(dis_judge, torch.ones_like(dis_judge)).cpu().data.numpy()

        reward = reward_cl + reward_d

        done = True


        self.count += 1

        # the nextState
        next_state = decoded.detach().cpu().data.numpy()
        self._state = decoded
        return next_state, reward, done, classifier_logits.cpu().data.numpy()
