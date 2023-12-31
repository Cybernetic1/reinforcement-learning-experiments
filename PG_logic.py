"""
This is the Logic Rules version, state vector is 9 propositions = 9 x 3 = 27-vector

Match the set of propositions to rules base.
The "head" of rules base is the same size as state vector = 27-dim.
The "tail" is the size of 1 proposition = 3-dim vector.
Retrieve the suggested action.

Need to get rid of policy gradient and use something else.

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

# reproducible (this may be an overkill... but it works)
seed=666
np.random.seed(seed)
torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
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

		self.memSize = 128

		# Episode data: actions, rewards:
		self.ep_actions = Variable(torch.Tensor())
		self.ep_rewards = []

		self._build_net()

		self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

	def net_info(self):
		# Total number of params:
		total = 0		# **** TO-DO
		return ("?x?x?", total)

	def _build_net(self):
		self.head = randn((27, self.memSize))
		self.tail = randn((3, self.memSize))
		self.softmax = nn.Softmax(dim=0)

	def forward(self, x):
		# input dim = n_features = 9 x 3 = 27
		# there are 9 h-networks each taking a dim-3 vector input
		# First we need to split the input into 9 parts:
		xs = torch.stack(torch.split(x, 3))
		print("xs =", xs)

		# 
		y = self.trm(xs)
		return y
		# What is the output here?  Old output = probs over actions
		# The most reasonable output is: probability distribution over actions.
		# But there is a waste of 3 dimensions
		# Perhaps an even better output format is: probability distribution over specific [x,y]'s.
		# Then we need to deal with the problem of merging duplicated [x,y] values.
		# The duplicated probabilities could be added or maxed.
		# P(A or B) = P(A) + P(B) - P(A and B)  but it's difficult to estimate P(A and B)
		# Non-monotonic logic is actually more economical as knowledge representation!
		# Another question is: if the rule-base suggests action X with probability P1,
		# and it also alternatively suggests the same action X with probability P2.
		# So it is actually up to us to interpret P1 and P2.
		# Naturally, we can normalize all P[i]'s to 1.
		# But then X would be chosen with probability P1 + P2.

	def choose_action(self, state):
		# Select an action (0-9) by running policy model and choosing based on the probabilities in state
		state = torch.from_numpy(state).type(torch.FloatTensor)
		y = self(Variable(state))
		# print("y =", y)
		# y = 27-dim vector = [probability, X, Y] x 9
		probs = self.softmax(torch.select(y, 1, 0))		# get every 1st element = probability
		# print("probs =", probs)
		c = Categorical(probs)
		i = c.sample()
		# print("i =", i)

		def discretize(z):
			if z > 0.5:
				return 2
			elif z < -0.5:
				return 0
			else:
				return 1

		action = discretize(y[i][1]) + discretize(y[i][2]) * 3
		print(action, end='')
		# **** Is it OK for "action" to be different from "i" (for back-prop) ?
		# Yes, as long as the reward (for action "i") is assigned correctly.
		# **** Explanation for failure of convergence:
		# "i" is real number, but "action" is discretized.
		# When i changes infinitesimally, "action" may remain unchanged.
		# Thus the reward fails to reflect changes in learned parameters.

		# Add log probability of our chosen action to our history
		# Unsqueeze(0): tensor (prob, grad_fn) ==> ([prob], grad_fn)
		log_prob = c.log_prob(i).unsqueeze(0)
		# print("log prob:", c.log_prob(action))
		# print("log prob unsqueezed:", log_prob)
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
		# print("policy history:", self.ep_actions)
		# print("rewards:", rewards)
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

	def save_net(self, fname):
		torch.save(self.state_dict(), "PyTorch_models/" + fname + ".dict")
		print("Model saved.")

	def load_net(self, fname):
		torch.load(self.state_dict(), fname + ".dict")
		print("Model loaded.")
