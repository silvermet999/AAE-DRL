import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os

from AAE import AAE_archi_opt
from DDPG import utils

cuda = True if torch.cuda.is_available() else False
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, discrete_features, max_action):
        super(Actor, self).__init__()
        self.discrete_features = discrete_features
        self.max_action = max_action

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 25),
            nn.ReLU(),
            nn.Linear(25, 25),
            nn.ReLU()
        )

        # Continuous action outputs (TD3 style)
        self.continuous_head = nn.Sequential(
            nn.Linear(25, 25),
            nn.ReLU(),
            nn.Linear(25, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        shared_features = self.shared(state)
        continuous_actions = self.max_action * self.continuous_head(shared_features)
        return continuous_actions


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, discrete_features):
        super(Critic, self).__init__()
        self.discrete_features = discrete_features

        # Calculate total discrete outputs
        total_discrete_outputs = sum(ranges for ranges in discrete_features.values())

        # Q1 Architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 25),
            nn.ReLU(),
            nn.Linear(25, 25),
            nn.ReLU(),
            nn.Linear(25, 1)  # TD3-style Q-value
        )

        # Q2 Architecture (Twin critic for TD3 part)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 25),
            nn.ReLU(),
            nn.Linear(25, 25),
            nn.ReLU(),
            nn.Linear(25, 1)  # TD3-style Q-value
        )

        # DDQN heads for discrete actions
        self.discrete_q = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(state_dim + action_dim, 25),
                nn.ReLU(),
                nn.Linear(25, 25),
                nn.ReLU(),
                nn.Linear(25, num_actions)  # Q-values for each discrete action
            ) for name, num_actions in discrete_features.items()
        })

    def forward(self, state, continuous_action):
        # Concatenate state and continuous action
        sa = torch.cat([state, continuous_action], 1)

        # Get TD3-style Q-values
        q1_cont = self.q1(sa)
        q2_cont = self.q2(sa)

        # Get DDQN-style Q-values for each discrete action space
        discrete_q_values = {
            name: self.discrete_q[name](sa)
            for name in self.discrete_ranges.keys()
        }

        return q1_cont, q2_cont, discrete_q_values

class TD3(object):
    def __init__(self, state_dim, action_dim, discrete_features, max_action):
        self.max_action = max_action
        self.continuous_dims = action_dim
        self.discrete_ranges = discrete_features

        # Initialize actor for continuous actions
        self.actor = Actor(state_dim, action_dim, discrete_features, max_action).cuda() if cuda else (
            Actor(state_dim, action_dim, discrete_features, max_action))
        self.actor_target = Actor(state_dim, action_dim, discrete_features, max_action).cuda() if cuda else (
            Actor(state_dim, action_dim, discrete_features, max_action))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        # Initialize critic
        self.critic = Critic(state_dim, action_dim, discrete_features).cuda() if cuda else (
            Critic(state_dim, action_dim, discrete_features))
        self.critic_target = Critic(state_dim, action_dim, discrete_features).cuda() if cuda else (
            Critic(state_dim, action_dim, discrete_features))
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # Initialize epsilon for epsilon-greedy exploration of discrete actions
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # TD3 parameters
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.policy_freq = 2
        self.total_it = 0
        self.replay_buffer = deque(maxlen=1000000)


    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).cuda() if cuda else torch.FloatTensor(state)

            # Get continuous actions from actor
            continuous_actions = self.actor(state_tensor)

            # Get Q-values for discrete actions
            sa = torch.cat([state_tensor, (continuous_actions.cuda() if cuda else continuous_actions)], 1)
            discrete_actions = {}

            # Epsilon-greedy selection for discrete actions
            for name, num_actions in self.discrete_ranges.items():
                if random.random() < self.epsilon:
                    discrete_actions[name] = random.randrange(num_actions)
                else:
                    q_values = self.critic.discrete_q[name](sa)
                    discrete_actions[name] = q_values.argmax().item()


            return continuous_actions, discrete_actions

    def train(self, batch_size=2):
        self.total_it += 1

        # Sample replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        batch_tensors = [(torch.FloatTensor(state),
                          continuous_action,
                          torch.tensor(list(discrete_actions.values())),
                          torch.FloatTensor(next_state),
                          torch.FloatTensor([reward]),
                          torch.FloatTensor([done]))
                         for state, continuous_action, discrete_actions, next_state, reward, done in batch]

        # Then unzip and stack
        state = torch.stack([b[0] for b in batch_tensors])
        continuous_action = torch.stack([b[1] for b in batch_tensors])
        discrete_actions = torch.stack([b[2] for b in batch_tensors])
        next_state = torch.stack([b[3] for b in batch_tensors])
        reward = torch.stack([b[4] for b in batch_tensors])
        done = torch.stack([b[5] for b in batch_tensors])

        # state, continuous_action, discrete_actions, next_state, reward, done = map(np.stack, zip(*batch))

        # Convert to tensors
        state = torch.FloatTensor(state).cuda() if cuda else torch.FloatTensor(state)
        continuous_action = continuous_action.cuda() if cuda else continuous_action
        next_state = torch.FloatTensor(next_state).cuda() if cuda else torch.FloatTensor(next_state)
        reward = torch.FloatTensor(reward).reshape(-1, 1).cuda() if cuda else torch.FloatTensor(reward).reshape(-1, 1)
        done = torch.FloatTensor(done).reshape(-1, 1).cuda() if cuda else torch.FloatTensor(done).reshape(-1, 1)

        with torch.no_grad():
            # Select next continuous actions with noise
            noise = torch.randn_like(continuous_action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            next_continuous_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute target Q values
            target_Q1, target_Q2, target_discrete_Q = self.critic_target(next_state, next_continuous_action)
            target_Q = torch.min(target_Q1, target_Q2)

            # Final targets for continuous Q values
            target_Q = reward + (1 - done) * 0.99 * target_Q

            # Compute target Q values for discrete actions
            discrete_targets = {}
            for name in self.discrete_ranges.keys():
                next_q_values = target_discrete_Q[name]
                next_q_value = next_q_values.max(dim=1, keepdim=True)[0]
                discrete_targets[name] = reward + (1 - done) * 0.99 * next_q_value

        # Get current Q estimates
        current_Q1, current_Q2, current_discrete_Q = self.critic(state, continuous_action)

        # Compute critic losses
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Add discrete action losses
        for name in self.discrete_ranges.keys():
            discrete_action_tensor = torch.LongTensor(discrete_actions[name]).cuda() if cuda else torch.LongTensor(discrete_actions[name])
            current_q = current_discrete_Q[name].gather(1, discrete_action_tensor.unsqueeze(1))
            critic_loss += F.mse_loss(current_q, discrete_targets[name])

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss (only for continuous actions)
            continuous_actions = self.actor(state)
            actor_loss = -self.critic.q1(state, continuous_actions).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def store_transition(self, state, continuous_action, discrete_actions, next_state, reward, done):
        self.replay_buffer.append((state, continuous_action, discrete_actions, next_state, reward, done))


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
