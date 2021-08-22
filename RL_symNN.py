"""
This is the symNN version, where the state vector is 9 propositions = 9 x 3 = 27-vector

Network topology = ...

============================================================
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
PyTorch: 1.9.0+cpu
gym: 0.8.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

# reproducible
np.random.seed(1)
torch.manual_seed(1)

class PolicyGradient(nn.Module):
	def __init__(
			self,
			n_actions,
			n_features,
			learning_rate=0.01,
			gamma=0.95,
	):
		super(PolicyGradient, self).__init__()
		self.n_actions = n_actions
		self.n_features = n_features

		self.lr = learning_rate
		self.gamma = gamma

		self.ep_obs, self.ep_as, self.ep_rs = [], [], []

		# Episode policy and reward history
		self.policy_history = Variable(torch.Tensor())
		self.reward_episode = []
		# Overall reward and loss history
		self.reward_history = []				# = ep_rs ?
		self.loss_history = []

		self._build_net()

		self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
		# self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

	def _build_net(self):
		# **** h-network, also referred to as "phi" in the literature
		# input dim = 3 because each proposition is a 3-vector
		self.h1 = nn.Linear(3, 8, bias=True)
		self.h2 = nn.Linear(8, self.n_actions, bias=True)

		# **** g-network, also referred to as "rho" in the literature
		# input dim can be arbitrary, here chosen to be n_actions
		self.g1 = nn.Linear(self.n_actions, self.n_actions + 3, bias=True)
		# output dim must be n_actions
		self.g2 = nn.Linear(self.n_actions + 3, self.n_actions, bias=True)

		# total number of weights = ...?

	def forward(self, x):
		x = self.h1(x)
		x = nn.ReLU(x)
		x = self.h2(x)
		x = nn.ReLU(x)
		x = self.g1(x)
		x = nn.ReLU(x)
		x = self.g2(x)
		x = nn.Softmax(x, dim=-1)
		return x

	def choose_action(self, state):
		#Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
		state = torch.from_numpy(state).type(torch.FloatTensor)
		state = self(Variable(state))
		c = Categorical(state)
		action = c.sample()

		# Add log probability of our chosen action to our history
		log_probs = c.log_prob(action).unsqueeze(0)
		if self.policy_history.dim() != 0:
			# print("log probs:", log_probs)
			self.policy_history = torch.cat([self.policy_history, log_probs])
		else:
			self.policy_history = (log_probs)
		return action

	def store_transition(self, s, a, r):		# state, action, reward
		self.ep_obs.append(s)
		self.ep_as.append(a)
		self.ep_rs.append(r)

	def learn(self):
		R = 0
		rewards = []

		# Discount future rewards back to the present using gamma
		# print("\nLength of reward episode:", len(self.ep_rs)) 
		for r in self.ep_rs[::-1]:			# [::-1] reverses a list
			R = r + self.gamma * R
			rewards.insert(0, R)

		# Scale rewards
		rewards = torch.FloatTensor(rewards)
		rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

		# Calculate loss
		# print("policy history:", self.policy_history)
		# print("rewards:", rewards)
		loss = (torch.sum(torch.mul(self.policy_history, Variable(rewards)).mul(-1), -1))

		# Update network weights
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		#Save and intialize episode history counters
		self.loss_history.append(loss.item())
		self.reward_history.append(np.sum(self.ep_rs))
		self.policy_history = Variable(torch.Tensor())
		self.reward_episode= []

		self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
		return rewards		# == discounted_ep_rs_norm

	def _discount_and_norm_rewards(self):
		# discount episode rewards
		discounted_ep_rs = np.zeros_like(self.ep_rs)
		running_add = 0
		for t in reversed(range(0, len(self.ep_rs))):
			running_add = running_add * self.gamma + self.ep_rs[t]
			discounted_ep_rs[t] = running_add

		# normalize episode rewards
		# print("discounted episode rewards=", discounted_ep_rs)
		discounted_ep_rs -= np.mean(discounted_ep_rs)
		discounted_ep_rs /= np.std(discounted_ep_rs)
		return discounted_ep_rs
