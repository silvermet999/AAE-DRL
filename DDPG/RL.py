import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os

cuda= False

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim+10, 12)
        self.l2 = nn.Linear(12, action_dim)

        self.max_action = max_action

    def forward(self, state, target):
        sa = torch.cat([state, target.cuda() if cuda else target], dim=1)

        a = F.relu(self.l1(sa))
        a = self.max_action * torch.tanh(self.l2(a))
        return a  # (values in range [-max_action, max_action])


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim + 10, 25)
        self.l2 = nn.Linear(25, 1)
        self.l3 = nn.Linear(state_dim + action_dim + 10, 25)
        self.l4 = nn.Linear(25, 1)

    def forward(self, state, action, target):
        sa = torch.cat([state, action, target], 1)

        q1 = F.relu(self.l1(sa))
        q1 = self.l2(q1)

        q2 = F.relu(self.l3(sa))
        q2 = self.l4(q2)
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
            # noise = noise.transpose(1,2)
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

if __name__ == "__main__":
    actor = Actor(state_dim=30, action_dim=1, max_action=10)
    critic = Critic(state_dim=30, action_dim=1)
    s = torch.FloatTensor(np.random.rand(5, 30))
    a = actor(s)
    n_s = critic(s, a)
    print('finished')
