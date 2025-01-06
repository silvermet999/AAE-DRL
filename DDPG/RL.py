import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os

cuda = True if torch.cuda.is_available() else False
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim+4, 22)
        self.l2 = nn.Linear(22, 18)
        self.l3 = nn.Linear(18, 14)
        self.l4 = nn.Linear(14, 10)
        self.l5 = nn.Linear(10, action_dim)

        self.max_action = max_action

    def forward(self, state, target):
        sa = torch.cat([state, target.cuda() if cuda else target], dim=1)

        a = F.relu(self.l1(sa))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        a = F.relu(self.l4(a))
        a = self.max_action * torch.tanh(self.l5(a))
        return a  # (values in range [-max_action, max_action])


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim + 4, 24)
        self.l2 = nn.Linear(24, 14)
        self.l3 = nn.Linear(14, 4)
        self.l4 = nn.Linear(4, 1)

        self.l5 = nn.Linear(state_dim + action_dim + 4, 24)
        self.l6 = nn.Linear(24, 14)
        self.l7 = nn.Linear(14, 4)
        self.l8 = nn.Linear(4, 1)

    def forward(self, state, action, target):
        sa = torch.cat([state, action, target], 1)

        q1 = F.relu(self.l1(sa))  # B x (state_dim + action_dim) ---> B x 256
        q1 = F.relu(self.l2(q1))  # B x 256 ---> B x 256
        q1 = F.relu(self.l3(q1))  # B x 256 ---> B x 1
        q1 = self.l4(q1)

        q2 = F.relu(self.l5(sa))  # B x (state_dim + action_dim) ---> B x 256
        q2 = F.relu(self.l6(q2))  # B x 256 ---> B x 256
        q2 = F.relu(self.l7(q2))  # B x 256 ---> B x 1
        q2 = self.l8(q2)
        return q1, q2

    def Q1(self, state, action, target):
        sa = torch.cat([state, action, target], 1)

        q1 = F.relu(self.l1(sa))
        q1 = self.l2(q1)
        return q1

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, lr=0.0, batch_size=1, discount=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.batch_size = batch_size
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.actor = Actor(state_dim, action_dim, max_action).to("cuda") if cuda else Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to("cuda") if cuda else Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.total_it = 0

    def select_action(self, state, target):
        state = torch.FloatTensor(state).to("cuda") if cuda else torch.FloatTensor(state)
        return self.actor(state, target).cpu().data.numpy()

    def train(self, replay_buffer):
        s, n_s, a, r, not_d, t = replay_buffer.sample()

        state = torch.FloatTensor(s).to("cuda") if cuda else torch.FloatTensor(s)
        next_state = torch.FloatTensor(n_s).to("cuda") if cuda else torch.FloatTensor(n_s)
        action = torch.FloatTensor(a).to("cuda") if cuda else torch.FloatTensor(a)
        reward = torch.FloatTensor(r).to("cuda") if cuda else torch.FloatTensor(r)
        done = torch.FloatTensor(1 - not_d).to("cuda") if cuda else torch.FloatTensor(1-not_d)
        target = torch.FloatTensor(t).to("cuda") if cuda else torch.FloatTensor(t)

        with torch.no_grad():
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (
                    self.actor_target(next_state, target) + noise
            ).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action, target)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action, target)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state, target), target).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


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
