"""
First multi-step experiment
RL will output some "intermediate" results that aren't actions.
actions 0-8 = tic-tac-toe actions
actions 9-17 = intermediate thoughts
These will be put into a special area of the "state".
For more explanations see: README-RL-with-autoencoder.md

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

class symNN(nn.Module):
	def __init__(self, input_dim, action_dim, hidden_dim, activation=F.relu, init_w=3e-3):
		super(symNN, self).__init__()

		# **** h-network, also referred to as "phi" in the literature
		# input dim = 2 because each proposition is a 2-vector
		self.h1 = nn.Linear(2, hidden_dim, bias=True)
		self.relu1 = nn.Tanh()
		self.h2 = nn.Linear(hidden_dim, 9, bias=True)
		self.relu2 = nn.Tanh()

		# **** g-network, also referred to as "rho" in the literature
		# input dim can be arbitrary, here chosen to be n_actions
		self.g1 = nn.Linear(9, hidden_dim, bias=True)
		self.relu3 = nn.Tanh()

		# output dim must be n_actions
		self.g2 = nn.Linear(hidden_dim, action_dim, bias=True)

	def forward(self, x):
		# input dim = n_features = 9 x 3 = 27
		# there are 9 h-networks each taking a dim-3 vector input
		# First we need to split the input into 9 parts:
		xs = torch.split(x, 2, dim=1)
		# print("xs=", xs)

		# h-network:
		ys = []
		for i in range(9 *2):					# repeat h1 9 *2 times
			ys.append( self.relu1( self.h1(xs[i]) ))
		zs = []
		for i in range(9 *2):					# repeat h2 9 *2 times
			zs.append( self.relu2( self.h2(ys[i]) ))

		# add all the z's together:
		z = torch.stack(zs, dim=1)
		z = torch.sum(z, dim=1)

		# g-network:
		z1 = self.g1(z)
		z1 = self.relu3(z1)
		z2 = self.g2(z1)
		# z2 = self.softmax(z2)
		return z2 # = logits

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

		hidden_dim = 16
		self.symnet = symNN(state_dim, action_dim, hidden_dim, activation=F.relu).to(device)

		self.q_criterion = nn.MSELoss()
		self.q_optimizer = optim.Adam(self.symnet.parameters(), lr=self.lr)

	def choose_action(self, state, deterministic=True):
		state = torch.FloatTensor(state).unsqueeze(0).to(device)

		logits = self.symnet(state)
		probs  = torch.softmax(logits, dim=1)
		dist   = Categorical(probs)
		action = dist.sample().numpy()[0]

		# print("chosen action=", action)
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

		logits = self.symnet(state)
		next_logits = self.symnet(next_state)

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
		config_h = "(2)-16-9"
		config_g = "9-16-(9)"
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
		return (config_h + ':' + config_g, total)

	def play_random(self, state, action_space):
		# Select an action (0-9) randomly
		# NOTE: random player never chooses occupied squares
		empties = [0,1,2,3,4,5,6,7,8]
		# Find and collect all empty squares
		# scan through all 9 propositions, each proposition is a 2-vector
		for i in range(0, 18, 2):
			# 'proposition' is a numpy array[3]
			proposition = state[i : i + 2]
			sym = proposition[0]
			if sym == 1 or sym == -1:
				x = proposition[1]
				j = x + 4
				empties.remove(j)
		# Select an available square randomly
		action = random.sample(empties, 1)[0]
		return action

	def save_net(self, fname):
		print("Model not saved.")

	def load_net(self, fname):
		print("Model not loaded.")
