import torch
import torch.utils.data
import os

import utils
from EnvClass import Env

import numpy as np

from utils import RL_dataloader
from RL import TD3
from AAE import AAE_archi_opt
from clfs import classifier

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

cuda = True if torch.cuda.is_available() else False

def evaluate_policy(policy, dataloader, env, episode_num=10):
    avg_reward = 0.
    env.reset()

    for i in range(0, episode_num):
        input, label = dataloader.next_data()
        obs = env.set_state(input)
        done = False
        episode_target = (label + torch.randint(4, label.shape)) % 4

        while not done:
            continuous_act, discrete_act = policy.select_action(obs)
            new_state, reward, done = env(continuous_act, discrete_act, episode_target)
            avg_reward += reward

    avg_reward /= episode_num

    return avg_reward




class Trainer(object):
    def __init__(self, train_loader, valid_loader, model_encoder, model_d, model_De, classifier, in_out, discrete):
        np.random.seed(5)
        torch.manual_seed(5)


        self.train_loader = RL_dataloader(train_loader)
        self.valid_loader = RL_dataloader(valid_loader)

        self.epoch_size = len(self.valid_loader)
        self.max_timesteps = 100000

        self.batch_size = 32
        self.eval_freq = 1000
        self.start_timesteps = 50
        self.max_episodes_steps = 1000000

        self.expl_noise = 0

        self.encoder = model_encoder
        self.D = model_d
        self.De = model_De
        self.classifier = classifier

        self.env = Env(self.encoder, self.D, self.De, self.classifier)
        self.replay_buffer = utils.ReplayBuffer()



        self.state_dim = in_out
        self.action_dim = 5
        self.discrete_features = discrete
        self.max_action = 1
        self.policy = TD3(self.state_dim, self.action_dim, self.discrete_features, self.max_action)



        self.continue_timesteps = 0
        self.save_models = True

        self.evaluations = []

    def train(self):
        sum_return = 0
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        state_t, label = self.train_loader.next_data()
        state = self.env.set_state(state_t)
        episode_target = (torch.randint(4, label.shape) + label) % 4

        done = False
        self.env.reset()

        for t in range(int(self.continue_timesteps), int(self.max_timesteps)):
            episode_timesteps += 1
            continuous_act, discrete_act = self.policy.select_action(state)

            next_state, reward, done = self.env(continuous_act, discrete_act, episode_target)

            self.policy.store_transition(state, continuous_act, discrete_act,
                          next_state, reward, done, episode_target)

            state = next_state
            episode_reward += reward

            if t >= self.start_timesteps:
                self.policy.train()

            if done:
                state_t, label = self.train_loader.next_data()
                episode_target = (torch.randint(4, label.shape) + label) % 4
                state = self.env.set_state(state_t)

                done = False
                self.env.reset()

                print('\repisode: {}, reward: {}'.format(episode_num + 1, episode_reward), end='')
                sum_return += episode_reward
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if (t + 1) % self.eval_freq == 0:
                episode_result = "episode: {} average reward: {}".format(episode_num,
                                                                                  sum_return / episode_num)
                print('\r' + episode_result)

                valid_episode_num = 6
                self.evaluations.append(evaluate_policy(self.policy, self.valid_loader, self.env,
                                                        episode_num=valid_episode_num))
                eval_result = "episodes: {}".format(self.evaluations[-1])
                print(eval_result)

in_out = 30
z_dim = 10
label_dim = 4

encoder_generator = AAE_archi_opt.EncoderGenerator(in_out, z_dim).cuda() if cuda else (
    AAE_archi_opt.EncoderGenerator(in_out, z_dim))
decoder = AAE_archi_opt.Decoder(z_dim, in_out, utils.discrete, utils.continuous, utils.binary)

discrete = {"state": 5,
            # "service": 13,
            "ct_state_ttl": 6,
            # "dttl": 9,
            # "sttl": 13,
            "trans_depth": 11,
            "proto": 2,
            "is_ftp_login":2
}


discriminator = AAE_archi_opt.Discriminator(z_dim, ).cuda() if cuda else (
    AAE_archi_opt.Discriminator(z_dim, ))
discriminator.eval()
classifier = classifier.classifier
classifier.eval()

train_loader, val_loader = utils.dataset_function(utils.dataset, 32, 32, train=True)

trainer = Trainer(train_loader, val_loader, encoder_generator, discriminator, decoder, classifier, in_out, discrete)
