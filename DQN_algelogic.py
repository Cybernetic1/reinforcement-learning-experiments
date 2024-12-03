"""
DQN (Deep Q Network) using my invention "algebraic logic network" to implement Q.
Board representation:  logic, dim-2
Code is adapted from DQN_shrink.py.
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
		states, actions, rewards, next_states, dones = \
			map(np.stack, zip(*batch)) # stack for each element
		'''
		the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
		zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
		the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
		np.stack((1,2)) => array([1, 2])
		'''
		# print("sampled state=", state)
		# print("sampled action=", action)
		return states, actions, rewards, next_states, dones

	def __len__(self):
		return len(self.buffer)

class AlgelogicNetwork(nn.Module):
	def __init__(self, input_dim, action_dim, hidden_size, activation=F.relu, init_w=3e-3):
		super(AlgelogicNetwork, self).__init__()

		# **** Define K predicates
		K = 16
		self.predicate = []
		for i in range(0,K):
			self.predicate[i].linear1 = nn.Linear(input_dim, hidden_size)
			self.predicate[i].linear2 = nn.Linear(hidden_size, hidden_size)
			self.predicate[i].logits_linear = nn.Linear(hidden_size, action_dim)
			self.predicate[i].logits_linear.weight.data.uniform_(-init_w, init_w)
			self.predicate[i].logits_linear.bias.data.uniform_(-init_w, init_w)

		# **** Define M rules
		M = 16
		self.ruleHead = []
		self.ruleTail = []
		for i in range(0,M):
			self.ruleHead[i] = torch.rand(K)
			self.ruleTail[i] = torch.rand(K)

		self.activation = F.relu

	# 首先定义什么是 x，及它是如何储存。
	# 它是 (point, predicate) pairs where predicate is just a number from {0...K}
	# size of state = W pairs.
	# 輸出的格式一樣
	# 計算方法：
	# for each rule:
	#	evaluate predicates on all points in x
	#	if rule matches, create output predicate
	def forward(self, state):
		# For each fact xi in x:
		for xi in x:
			# First, evaluate all predicates
			for j in range(0,K)		# for each predicate
				y = self.activation(self.predicate[j].linear1(xi))
				y = self.activation(self.predicate[j].linear2(y))
				y = self.activation(self.predicate[j].linear3(y))
				# keep truth values for later
				# if y ~= 1 the predicate is true, but it is true for P(xi) only

		for i in range(0,M)			# for each rule
			t = 1.0
			soft top-k self.ruleHead[i] select k predicates
			match = multiply truth values of all K predicates
			self.ruleTail[i] is a distribution over K predicates, multiply by match
			= output distribution of rule i
			exp to calculate probability distribution
		return prob distro for all M rules

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

		hidden_dim = 9
		self.qnet = QNetwork(state_dim, action_dim, hidden_dim, activation=F.relu).to(device)

		self.q_criterion = nn.MSELoss()
		self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

	def choose_action(self, state, deterministic=True):
		state = torch.FloatTensor(state).unsqueeze(0).to(device)

		logits = self.qnet(state)
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

		logits = self.qnet(state)
		next_logits = self.qnet(next_state)

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
		config = "(9)-9-9-(9)"
		neurons = config.split('-')
		last_n = 9
		total = 0
		for n in neurons[1:-1]:
			n = int(n)
			total += last_n * n
			last_n = n
		total += last_n * 9
		return (config, total)

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
