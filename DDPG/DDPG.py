import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

cuda = True if torch.cuda.is_available() else False


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()
		self.max_action = max_action
		seq = [nn.Linear(state_dim, 400),
			   nn.ReLU(),
			   nn.Linear(400, 400),
			   nn.ReLU(),
			   nn.Linear(400, 300),
			   nn.ReLU(),
			   nn.Linear(300, action_dim)]
		# self.l1 = nn.Linear(state_dim, 400)
		# self.l2 = nn.Linear(400, 400)
		# self.l2_additional = nn.Linear(400, 300)
		# self.l3 = nn.Linear(300, action_dim)
		self.seq = nn.Sequential(*seq)


	
	def forward(self, x):
		# x = F.relu(self.l1(x))
		# x = F.relu(self.l2(x))
		# x = F.relu(self.l2_additional(x))
		x = self.max_action * torch.tanh(self.seq(x))
		return x 


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400 + action_dim, 300)
		self.l3_additional = nn.Linear(300, 300)
		self.l3 = nn.Linear(300, 1)


	def forward(self, x, u):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(torch.cat([x, u], 1)))
		x = self.l3_additional(x)
		x = self.l3(x)
		return x


class DDPG(object):
	def __init__(self, state_dim, action_dim, max_action):
		self.actor = Actor(state_dim, action_dim, max_action).cuda() if cuda else Actor(state_dim, action_dim, max_action)
		self.actor_target = Actor(state_dim, action_dim, max_action).cuda() if cuda else Actor(state_dim, action_dim, max_action)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

		self.critic = Critic(state_dim, action_dim).cuda() if cuda else Critic(state_dim, action_dim)
		self.critic_target = Critic(state_dim, action_dim).cuda() if cuda else Critic(state_dim, action_dim)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).cuda() if cuda else torch.FloatTensor(state.reshape(1, -1))
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations, batch_size=32, discount=0.99, tau=0.001):

		for it in range(iterations):

			# Sample replay buffer
			x, y, u, r, d = replay_buffer.sample(batch_size)
			state = torch.FloatTensor(x).cuda() if cuda else torch.FloatTensor(x)
			action = torch.FloatTensor(u).cuda() if cuda else torch.FloatTensor(u)
			next_state = torch.FloatTensor(y).cuda() if cuda else torch.FloatTensor(y)
			done = torch.FloatTensor(1 - d).cuda() if cuda else torch.FloatTensor(1 - d)
			reward = torch.FloatTensor(r).cuda() if cuda else torch.FloatTensor(r)

			# Compute the target Q value
			target_Q = self.critic_target(next_state, self.actor_target(next_state))
			target_Q = reward + (done * discount * target_Q).detach()

			# Get current Q estimate
			current_Q = self.critic(state, action)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q, target_Q)

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Compute actor loss
			actor_loss = -self.critic(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

