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
			learning_rate=0.001,
			gamma=0.9,		# only for discounting within a game (episode)
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
		# input dim = n_features = 9 x 3 = 27
		# there are 9 h-networks each taking a dim-3 vector input
		# First we need to split the input into 9 parts:
		xs = torch.split(x, 3)

		# h-network:
		ys = []
		relu1 = nn.ReLU()
		for i in range(9):							# repeat h1 9 times
			ys.append( relu1( self.h1(xs[i]) ))
		zs = []
		relu2 = nn.ReLU()
		for i in range(9):							# repeat h2 9 times
			zs.append( relu2( self.h2(ys[i]) ))

		# add all the z's together:
		z = torch.stack(zs, dim=1)
		z = torch.sum(z, dim=1)

		# g-network:
		z1 = self.g1(z)
		relu3 = nn.ReLU()
		z1 = relu3(z1)
		z2 = self.g2(z1)
		softmax = nn.Softmax(dim=0)
		z2 = softmax(z2)
		return z2

	def choose_action(self, state):
		#Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
		state = torch.from_numpy(state).type(torch.FloatTensor)
		probs = self(Variable(state))
		c = Categorical(probs)
		action = c.sample()

		# Add log probability of our chosen action to our history
		# Unsqueeze(0): tensor (prob, grad_fn) ==> ([prob], grad_fn)
		log_prob = c.log_prob(action).unsqueeze(0)
		# print("log prob:", c.log_prob(action))
		# print("log prob unsqueezed:", log_prob)
		if self.policy_history.dim() != 0:
			self.policy_history = torch.cat([self.policy_history, log_prob])
		else:
			self.policy_history = (log_prob)
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
