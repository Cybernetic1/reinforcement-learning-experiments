"""
This is the symNN version, where the state vector is 9 propositions = 9 x 3 = 27-vector

Refer to net_config() below for the current network topology and total number of weights info.

For example: h = (3-9-9) x 9, g = (9-12-9) x 1
Total # weights = (3 * 9 + 9 * 9) * 9 + 9 * 9 + 9 * 9 = 1134
Duplicate weights are counted because they are updated multiple times.

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

# reproducible (this may be an overkill...)
seed=666
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

class PolicyGradient(nn.Module):
	def __init__(
			self,
			n_actions,
			n_features,
			learning_rate,
			gamma,		# only for discounting within a game (episode), seems useless
	):
		super(PolicyGradient, self).__init__()
		self.n_actions = n_actions
		self.n_features = n_features

		self.lr = learning_rate
		self.gamma = gamma

		# Episode data: actions, rewards:
		self.ep_actions = Variable(torch.Tensor())
		self.ep_rewards = []

		self._build_net()

		self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

	def net_info(self):
		config_h = "(3)-10-8"
		config_g = "8-(9)"
		total = 0
		neurons = config_h.split('-')
		last_n = 3
		for n in neurons[1:]:
			n = int(n)
			total += last_n * n
			last_n = n
		total *= 9

		neurons = config_g.split('-')
		for n in neurons[1:-1]:
			n = int(n)
			total += last_n * n
			last_n = n
		total += last_n * 9
		return (config_h + 'x' + config_g, total)

	def _build_net(self):
		# **** h-network, also referred to as "phi" in the literature
		# input dim = 3 because each proposition is a 3-vector
		self.h1 = nn.Linear(3, 9, bias=True)
		self.relu1 = nn.Tanh()
		self.h2 = nn.Linear(9, 9, bias=True)
		self.relu2 = nn.Tanh()

		# **** g-network, also referred to as "rho" in the literature
		# input dim can be arbitrary, here chosen to be n_actions
		self.g1 = nn.Linear(9, 9, bias=True)
		self.relu3 = nn.Tanh()

		# output dim must be n_actions
		self.g2 = nn.Linear(9, self.n_actions, bias=True)
		self.softmax = nn.Softmax(dim=0)

	def forward(self, x):
		# input dim = n_features = 9 x 3 = 27
		# there are 9 h-networks each taking a dim-3 vector input
		# First we need to split the input into 9 parts:
		xs = torch.split(x, 3)

		# h-network:
		ys = []
		for i in range(9):						# repeat h1 9 times
			ys.append( self.relu1( self.h1(xs[i]) ))
		zs = []
		for i in range(9):						# repeat h2 9 times
			zs.append( self.relu2( self.h2(ys[i]) ))

		# add all the z's together:
		z = torch.stack(zs, dim=1)
		z = torch.sum(z, dim=1)

		# g-network:
		z1 = self.g1(z)
		z1 = self.relu3(z1)
		z2 = self.g2(z1)
		z2 = self.softmax(z2)
		return z2

	def choose_action(self, state):
		#Select an action (0-9) by running policy model and choosing based on the probabilities in state
		state = torch.from_numpy(state).type(torch.FloatTensor)
		probs = self(Variable(state))
		c = Categorical(probs)
		action = c.sample()

		# Add log probability of our chosen action to our history
		# Unsqueeze: returns a new tensor with a dimension of size 1 inserted at the specified position.
		# Unsqueeze(0): tensor (prob, grad_fn) ==> ([prob], grad_fn)
		log_prob = c.log_prob(action).unsqueeze(0)
		print("log prob:", c.log_prob(action))
		print("log prob unsqueezed:", log_prob)
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
			x = action % 3
			y = action // 3
			occupied = False
			for i in range(0, 27, 3):		# scan through all 9 propositions, each proposition is a 3-vector
				# 'proposition' is a numpy array[3]
				proposition = state[i : i + 3]
				# print("proposition=",proposition)
				if ([x,y,1] == proposition).all():
					occupied = True
					break
				if ([x,y,-1] == proposition).all():
					occupied = True
					break
			if not occupied:
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
			#rewards = torch.FloatTensor([-1.0])
		#else:
		rewards = torch.FloatTensor(rewards)
			#rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
		# print(rewards)

		# Calculate loss
		print("policy history:", self.ep_actions)
		print("rewards:", rewards)
		# loss = torch.sum(torch.mul(self.ep_actions, Variable(rewards)).mul(-1), -1)
		loss = sum(torch.mul(self.ep_actions, Variable(rewards)).mul(-1), -1)

		# Update network weights
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# Empty episode data
		self.ep_actions = Variable(torch.Tensor())
		self.ep_rewards = []
		return rewards		# == discounted_ep_rewards_norm

	def clear_data(self):
		# empty episode data
		self.ep_actions = Variable(torch.Tensor())
		self.ep_rewards = []

	def save_net(self, fname):
		torch.save(self.state_dict(), "PyTorch_models/" + fname + ".dict")
		print("Model saved.")

	def load_net(self, fname):
		torch.load(self.state_dict(), fname + ".dict")
		print("Model loaded.")
