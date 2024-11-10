# libraries
import torch
import torch.nn as nn
import os
import numpy as np


class Env(nn.Module):
    def __init__(self, model_G, model_D):
        super(Env, self).__init__()

        self._state = None
        self.generator = model_G
        self.disciminator = model_D

        self.d_reward_coeff =1
        self.cl_reward_coeff =30

        # for calculating the discriminator reward
        self.hinge = torch.nn.HingeEmbeddingLoss()

        self.count = 0
        # self.save_path = os.path.join(args.result_dir, 'RL_train')
        # os.makedirs(self.save_path, exist_ok=True)

    def reset(self):
        self.count = 0

    def set_state(self, state):
        self._state = state
        return state.detach().cpu().numpy()

    def forward(self, action, episode_target, save_fig=False, t=None):
        # episodeTarget: the number that the RL agent is trying to find
        with (torch.no_grad()):
            z = action.to("cuda") if torch.cuda.is_available() else action
            gen_out = self.generator(z)
            dis_judge = self.disciminator(gen_out)

        batch_size = len(episode_target)
        episode_target = episode_target.to(torch.int64)
        # reward_cl = self.cl_reward_coeff * np.exp(gen_sample[1, episode_target].cpu().data.numpy())
        reward_cl = - self.d_reward_coeff * self.hinge(dis_judge, -1 * torch.ones_like(dis_judge)).cpu().data.numpy()
        reward = reward_cl

        done = True


        self.count += 1

        # the nextState
        next_state = gen_out.detach()
        self._state = gen_out
        return next_state, reward, done, dis_judge.cpu().data.numpy()