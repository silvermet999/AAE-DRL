import torch
import torch.utils.data
import torch.nn.parallel

from EnvClass import Env

import numpy as np
import os

from utils import ReplayBuffer, RL_dataloader
from RL import TD3
# from torch.utils.tensorboard import SummaryWriter
import pickle

cuda = False

def evaluate_policy(policy, dataloader, env, episode_num=10, t=None):
    avg_reward = 0.
    env.reset()

    for i in range(0, episode_num):
        input, label = dataloader.next_data()
        obs = env.set_state(input)
        done = False
        episodeTarget = (label + torch.randint(1, label.shape))
        while not done:
            action = policy.select_action(np.array(obs), episodeTarget)
            action = torch.tensor(action)
            new_state, reward, done, _ = env(action, episodeTarget, t)
            avg_reward += reward

    avg_reward /= episode_num

    return avg_reward


class Trainer(object):
    def __init__(self, train_loader, valid_loader, model_encoder, model_d, model_De):
        np.random.seed(5)
        torch.manual_seed(5)

        self.train_loader = RL_dataloader(train_loader)
        self.valid_loader = RL_dataloader(valid_loader)

        self.epoch_size = len(self.valid_loader)
        self.max_timesteps = 100000

        self.batch_size = 32
        self.batch_size_actor = 32
        self.eval_freq = 1000
        # self.save_models = args.save_models
        self.start_timesteps = 50
        self.max_episodes_steps = 1000000

        self.z_dim = 12
        self.max_action = 1
        self.expl_noise = 0.2

        self.encoder = model_encoder
        self.D = model_d
        self.De = model_De

        self.env = Env(self.encoder, self.D, self.De)

        self.state_dim = 27
        self.action_dim = 12
        self.max_action = 1

        self.policy = TD3(self.state_dim, self.action_dim, self.max_action, 0.0005, self.batch_size_actor,
                          0.99, 0.005, 0.2, 0.5, 2)

        self.replay_buffer = ReplayBuffer()

        # self.model_path = os.path.join(args.model_dir, 'RL_train')
        # os.makedirs(self.model_path, exist_ok=True)
        #
        # self.rng_path = os.path.join(self.model_path, 'state.pkl')

        self.continue_timesteps = 0
        # if load_model:
        #     self.continue_timesteps = int(args.model_name.split('_')[-1])
        #     self.replay_buffer.load()
        #     self.policy.load(args.model_name, directory=self.model_path)
        #     with open(self.rng_path, 'rb') as f:
        #         np_rgn_state, torch_rng_state = pickle.load(f)
        #         np.random.set_state(np_rgn_state)
        #         torch.set_rng_state(torch_rng_state)

        # self.log_path = os.path.join(args.log_dir, 'RL_train')
        # os.makedirs(self.log_path, exist_ok=True)
        # self.writer = SummaryWriter(self.log_path)

        self.evaluations = []

    def train(self):
        """
        train RL
        """
        sum_return = 0
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        # get state and corresponding target value
        state_t, label = self.train_loader.next_data()
        state = self.env.set_state(state_t)
        episode_target = (torch.randint(10, label.shape) + label) % 10

        done = False
        self.env.reset()

        print('start/continue model from t: {}'.format(self.continue_timesteps))
        print('start buffer length: {}'.format(len(self.replay_buffer)))
        for t in range(int(self.continue_timesteps), int(self.max_timesteps)):
            episode_timesteps += 1
            # Select action randomly or according to policy
            if t < self.start_timesteps:
                # action_t = torch.FloatTensor(self.batch_size, self.z_dim).uniform_(-self.max_action, self.max_action)
                action_t = torch.randn(self.batch_size, self.z_dim)
                action = action_t.detach().cpu().numpy()
            else:
                action = (
                        self.policy.select_action(state, episode_target)
                        + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
                ).clip(-self.max_action, self.max_action)
                # action = self.policy.select_action(state)
                action = np.float32(action)
                action_t = torch.tensor(action).to("cuda") if cuda else torch.tensor(action)

            # Perform action
            next_state, reward, done, _ = self.env(action_t, episode_target)

            # Store data in replay buffer
            self.replay_buffer.add((state, next_state, action, reward, done, episode_target))

            state = next_state
            episode_reward += reward


            # Train agent after collecting sufficient data
            if t >= self.start_timesteps:
                self.policy.train(self.replay_buffer)

            if done:
                # Reset environment
                state_t, label = self.train_loader.next_data()
                episode_target = (torch.randint(10, label.shape) + label) % 10
                state = self.env.set_state(state_t)

                done = False
                self.env.reset()

                print('\rstep: {}, episode: {}, reward: {}'.format(t + 1, episode_num + 1, episode_reward), end='')
                # self.writer.add_scalar("episode_reward", episode_reward, t + 1)
                sum_return += episode_reward
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if (t + 1) % self.eval_freq == 0:
                episode_result = "step: {} episode: {} average reward: {}".format(t + 1, episode_num,
                                                                                  sum_return / episode_num)
                print('\r' + episode_result)

                valid_episode_num = 6
                self.evaluations.append(evaluate_policy(self.policy, self.valid_loader, self.env,
                                                        episode_num=valid_episode_num, t=t))
                eval_result = "evaluation over {} episodes: {}".format(valid_episode_num, self.evaluations[-1])
                print(eval_result)

                # if self.save_models:
                #     with open(self.rng_path, 'wb') as f:
                #         pickle.dump((np.random.get_state(), torch.get_rng_state()), f, -1)
                #     print('\rsaving model RL_{}'.format(t + 1), end='')
                #     self.policy.save('RL_{}'.format(t + 1), directory=self.model_path)
                #     print('\rsaved model RL_{}'.format(t + 1), end='')
                #
                #     print('\rsaving replay buffer', end='')
                #     # save some of environment buffer seen so far
                #     self.replay_buffer.save()
                #     print('\rsaved replay buffer', end='')

                # print('\rsaving replay buffer samples', end='')
                # self.replay_buffer.save_samples(len(self.replay_buffer) * 0.01, self.decoder)
                # print('\rsaved replay buffer samples', end='')

                # with open(os.path.join(self.log_path, "logs.txt"), "a") as f:
                #     f.write(episode_result + '\n' + eval_result + '\n\n')
                #     f.close()

                print('\rfinished saving', end='')
                print('\r---------------------------------------')
