"""
Deep Q Network
modified from Q-table where the table is replaced by a deep NN.

Using:
PyTorch: 1.9.1+cpu
gym: 0.8.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.distributions import Normal

import random
import numpy as np
np.random.seed(7)
torch.manual_seed(7)
device = torch.device("cpu")

class ReplayBuffer:
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []
		self.position = 0

	def push(self, state, action, reward, next_state, done):
		if len(self.buffer) < self.capacity:
			self.buffer.append(None)
		self.buffer[self.position] = (state, action, reward, next_state, done)
		self.position = (self.position + 1) % self.capacity

	def last_reward(self):
		return self.buffer[self.position-1][2]

	def sample(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		state, action, reward, next_state, done = \
			map(np.stack, zip(*batch)) # stack for each element
		'''
		the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
		zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
		the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
		np.stack((1,2)) => array([1, 2])
		'''
		# print("sampled state=", state)
		# print("sampled action=", action)
		return state, action, reward, next_state, done

	def __len__(self):
		return len(self.buffer)

class DQN():

	def __init__(
			self,
			action_dim,
			state_dim,
			learning_rate = 3e-4,
			gamma = 0.9 ):
		super(DQN, self).__init__()

		self.action_dim = action_dim
		self.state_dim = state_dim
		self.lr = learning_rate
		self.gamma = gamma

		self.replay_buffer = ReplayBuffer(int(1e6))

		self._build_net()

		self.q_criterion = nn.MSELoss()
		self.q_optimizer = optim.Adam(self.trm.parameters(), lr=self.lr)

	def _build_net(self):
		encoder_layer = nn.TransformerEncoderLayer(d_model=3, nhead=1)
		self.trm = nn.TransformerEncoder(encoder_layer, num_layers=3)
		# W is a 3x9 matrix, to convert 3-vector to 9-vector probability distribution:
		self.W = Variable(torch.randn(3, 9), requires_grad=True)
		self.softmax = nn.Softmax(dim=0)

	def forward(self, x):
		# input dim = n_features = 9 x 3 = 27
		# First we need to split the input into 9 parts:
		# print("x =", x)
		xs = torch.stack(torch.split(x, 3, 1), 1)
		# print("xs =", xs)
		# There is a question of how these are stacked, 9x3 or 3x9?
		# it has to conform with Transformer's d_model = 3
		ys = self.trm(xs) # no need to split results, already in 9x3 chunks
		# print("ys =", ys)
		# it seems that only the last 3-dim vector is useful
		u = torch.matmul( ys.select(1, 8), self.W )
		# *** sum the probability distributions together:
		# z = torch.stack(zs, dim=1)
		# u = torch.sum(z, dim=1)
		# v = self.softmax(u)
		# print("v =", v)
		return u

	def choose_action(self, state, deterministic=True):
		# Select an action (0-8) by running policy model and choosing based on the probabilities in state
		state = torch.from_numpy(state).type(torch.FloatTensor)
		logits = self.forward(Variable(state).unsqueeze(0))[0]
		probs = self.softmax(logits)
		# probs = 9-dim vector
		# print("probs =", probs)
		distro = Categorical(probs)
		action = distro.sample().numpy()
		# print("(" + str(action), end=')')

		# Add log probability of our chosen action to our history
		# Unsqueeze(0): tensor (prob, grad_fn) ==> ([prob], grad_fn)
		# log_prob = distro.log_prob(action).unsqueeze(0)
		# print("log prob:", c.log_prob(action))
		# print("log prob unsqueezed:", log_prob)

		return action

	def update(self, batch_size, reward_scale, gamma=0.99):
		alpha = 1.0  # trade-off between exploration (max entropy) and exploitation (max Q)

		state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
		# print('sample (state, action, reward, next state, done):', state, action, reward, next_state, done)

		state      = torch.FloatTensor(state).to(device)
		next_state = torch.FloatTensor(next_state).to(device)
		action     = torch.LongTensor(action).to(device)
		reward     = torch.FloatTensor(reward).to(device) # .to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
		done       = torch.BoolTensor(done).to(device)

		logits = self.forward(state)	# state dim = 512x27
		next_logits = self.forward(next_state)
		# print("logits:", logits.shape)

		# **** Train deep Q function, this is just Bellman equation:
		# DQN(st,at) += η [ R + γ max_a DQN(s_t+1,a) - DQN(st,at) ]
		# DQN[s, action] += self.lr *( reward + self.gamma * np.max(DQN[next_state, :]) - DQN[s, action] )
		# max 是做不到的，但似乎也可以做到。 DQN 输出的是 probs.
		# probs 和 Q 有什么关系？  Q 的 Boltzmann 是 probs (SAC 的做法).
		# This implies that Q = logits.
		# logits[at] += self.lr *( reward + self.gamma * np.max(logits[next_state, next_a]) - logits[at] )
		q = logits[range(logits.shape[0]), action]
		m = torch.max(next_logits, 1, keepdim=False).values
		# print("m:", m.shape)
		# q = q + self.lr *( reward + self.gamma * m - q )
		target_q = torch.where(done, reward, reward + self.gamma * m)
		# print("q, target_q:", q.shape, target_q.shape)
		q_loss = self.q_criterion(q, target_q.detach())

		self.q_optimizer.zero_grad()
		q_loss.backward()
		self.q_optimizer.step()

		return

	def net_info(self):
		config = "(27)-4L-(27)"
		return (config, None)

	def play_random(self, state, action_space):
		# NOTE: random player never chooses occupied squares
		empties = [0,1,2,3,4,5,6,7,8]
		# Find and collect all empty squares
		# scan through all 9 propositions, each proposition is a 3-vector
		for i in range(0, 27, 3):
			# 'proposition' is a numpy array[3]
			proposition = state[i : i + 3]
			sym = proposition[2]
			if sym == 1 or sym == -1:
				x = proposition[0]
				y = proposition[1]
				j = y * 3 + x
				empties.remove(j)
		# Select an available square randomly
		action = random.sample(empties, 1)[0]
		return action

	def save_net(self, fname):
		print("Model not saved.")

	def load_net(self, fname):
		print("Model not loaded.")
