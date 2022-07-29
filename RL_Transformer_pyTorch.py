"""
This is the Transformer version, where the state vector is 9 propositions = 9 x 3 = 27-vector

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
		self.ep_as = Variable(torch.Tensor())
		self.ep_rs = []

		self._build_net()

		self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

	def net_info(self):
		# Total number of params:
		total = 3*4*3
		return ("3x4x3", total)

	def _build_net(self):
		encoder_layer = nn.TransformerEncoderLayer(d_model=3, nhead=4)
		self.tr = nn.TransformerEncoder(encoder_layer, num_layers=3)

	def forward(self, x):
		# input dim = n_features = 9 x 3 = 27
		# there are 9 h-networks each taking a dim-3 vector input
		# First we need to split the input into 9 parts:
		xs = torch.split(x, 3)

		out = self.tr(xs)
		return out
		# What is the output here?  Old output = probs over actions
		# The most reasonable output is: probability distribution over actions.
		# But there is a waste of 3 dimensions
		# Perhaps an even better output format is: probability distribution over specific [x,y]'s.
		# Then we need to deal with the problem of merging duplicated [x,y] values.
		# The duplicated probabilities could be added or maxed.

	def choose_action(self, state):
		# Select an action (0-9) by running policy model and choosing based on the probabilities in state
		state = torch.from_numpy(state).type(torch.FloatTensor)
		acts = self(Variable(state))		# check: dim(acts) = 9
		print("dim acts =", len(acts))
		action_xy = random.choice(acts)		# action in [x,y,player] format
		action = action_xy[0] + action_xy[1] * 3

		# Add log probability of our chosen action to our history
		# Unsqueeze(0): tensor (prob, grad_fn) ==> ([prob], grad_fn)
		log_prob = c.log_prob(action).unsqueeze(0)
		# print("log prob:", c.log_prob(action))
		# print("log prob unsqueezed:", log_prob)
		if self.ep_as.dim() != 0:
			self.ep_as = torch.cat([self.ep_as, log_prob])
		else:
			self.ep_as = (log_prob)
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
		#if len(rewards) == 1:
			#rewards = torch.FloatTensor([-1.0])
		#else:
		rewards = torch.FloatTensor(rewards)
			#rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
		# print(rewards)

		# Calculate loss
		# print("policy history:", self.ep_as)
		# print("rewards:", rewards)
		# loss = torch.sum(torch.mul(self.ep_as, Variable(rewards)).mul(-1), -1)
		loss = sum(torch.mul(self.ep_as, Variable(rewards)).mul(-1), -1)

		# Update network weights
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# Empty episode data
		self.ep_as = Variable(torch.Tensor())
		self.ep_rs = []
		return rewards		# == discounted_ep_rs_norm

	def save_net(self, fname):
		torch.save(self.state_dict(), "PyTorch_models/" + fname + ".dict")
		print("Model saved.")

	def load_net(self, fname):
		torch.load(self.state_dict(), fname + ".dict")
		print("Model loaded.")
