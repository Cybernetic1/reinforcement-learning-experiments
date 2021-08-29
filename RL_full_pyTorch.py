"""
Fully-connected version, where state vector is a 3 x 3 = 9-vector

Refer to net_config() below for the current network topology and # of weights info.

For example: (9 inputs)-16-16-16-16-(9 outputs)
Total num of weights = 9 * 16 * 2 + 16 * 16 * 3 = 1056
We want num of weights to be close to that of symNN = 1080

============================================================
Policy Gradient, Reinforcement Learning.  Adapted from:
Morvan Zhou's tutorial page: https://morvanzhou.github.io/tutorials/

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

		# Episode policy
		self.policy_history = Variable(torch.Tensor())

		self._build_net()

		self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

	def net_info(self):
		config = "(9)-16-16-16-16-(9)"
		neurons = config.split('-')
		last_n = 9
		total = 0
		for n in neurons[1:-1]:
			n = int(n)
			total += last_n * n
			last_n = n
		total += last_n * 9
		return (config, total)

	def _build_net(self):
		self.l1 = nn.Linear(self.n_features, 16, bias=True)
		self.l2 = nn.Linear(16, 16, bias=True)
		self.l3 = nn.Linear(16, 16, bias=True)
		self.l4 = nn.Linear(16, 16, bias=True)
		self.l5 = nn.Linear(16, self.n_actions, bias=False)

	def forward(self, x):
		model = torch.nn.Sequential(
			self.l1,
			# nn.Dropout(p=0.6),
			nn.ReLU(),
			self.l2,
			nn.ReLU(),
			self.l3,
			nn.ReLU(),
			self.l4,
			nn.ReLU(),
			self.l5,
			nn.Softmax(dim=-1),
			)
		return model(x)

	def choose_action(self, state):
		#Select an action (0-8) by running policy model and choosing based on the probabilities in state
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

	def play_random(self, state, action_space):
		# Select an action (0-9) randomly
		# NOTE: random player never chooses occupied squares
		while True:
			action = action_space.sample()
			if state[action] == 0:
				break
		return action

	def store_transition(self, s, a, r):	# state, action, reward
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
		self.policy_history = Variable(torch.Tensor())

		self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
		return rewards		# == discounted_ep_rs_norm

	def save_net(self, fname):
		torch.save(self.state_dict(), fname + ".dict")
		print("Model saved.")
