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
np.random.seed(7)
torch.manual_seed(7)

class PolicyGradient(nn.Module):
	def __init__(
			self,
			n_actions,
			n_features,
			learning_rate = 0.001,
			gamma = 0.9,
	):
		super(PolicyGradient, self).__init__()
		self.n_actions = n_actions
		self.n_features = n_features

		self.lr = learning_rate
		self.gamma = gamma

		self.ep_rewards = []
		# Episode policy
		self.ep_actions = Variable(torch.Tensor())

		self._build_net()

		self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

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

		self.model = torch.nn.Sequential(
			self.l1,
			# nn.Dropout(p=0.6),
			# nn.ReLU(),
			nn.Tanh(),
			self.l2,
			nn.Tanh(),
			self.l3,
			nn.Tanh(),
			self.l4,
			nn.Tanh(),
			self.l5,
			nn.Softmax(dim=-1),
			)

	def forward(self, x):
		return self.model(x)

	def choose_action(self, state):
		#Select an action (0-8) by running policy model and choosing based on the probabilities
		state = torch.from_numpy(state).type(torch.FloatTensor)
		probs = self(Variable(state))
		# print("probs =", probs)
		# action = torch.argmax(probs)
		c = Categorical(probs)
		action = c.sample()
		# print("action =", action)

		# log probability of our chosen action
		log_prob = c.log_prob(action).unsqueeze(0)
		# print("log prob:", log_prob)
		if self.ep_actions.dim() != 0:
			self.ep_actions = torch.cat([self.ep_actions, log_prob])
		else:
			self.ep_actions = (log_prob)
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
		# s is not needed, a is stored during choose_action().
		self.ep_rewards.append(r)

	def learn(self):
		R = 0
		rewards = []

		# Discount future rewards back to the present using gamma
		# print("\nLength of reward episode:", len(self.ep_rewards)) 
		for r in self.ep_rewards[::-1]:			# [::-1] reverses a list
			R = r + self.gamma * R
			rewards.insert(0, R)

		# Scale rewards
		#if len(rewards) == 1:
		#	rewards = torch.FloatTensor([0])
		#else:
		rewards = torch.FloatTensor(rewards)
		#	rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

		# Calculate loss
		# print("policy history:", self.ep_actions)
		# print("rewards:", rewards)
		loss = (torch.sum(torch.mul(self.ep_actions, Variable(rewards)).mul(-1), -1))

		# Update network weights
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# empty episode data
		self.ep_actions = Variable(torch.Tensor())
		self.ep_rewards = []
		return

	def clear_data(self):
		# empty episode data
		self.ep_actions = Variable(torch.Tensor())
		self.ep_rewards = []

	def save_net(self, fname):
		torch.save(self.state_dict(), "PyTorch_models/" + fname + ".dict")
		print("Model saved.")

	def load_net(self, fname):
		model = PolicyGradient(9, 9)
		model.load_state_dict(torch.load("PyTorch_models/" + fname + ".dict"))
		model.eval()
		print("Model loaded.")
