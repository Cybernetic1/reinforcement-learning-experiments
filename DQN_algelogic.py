"""
DQN (Deep Q Network) using my invention "algebraic logic network" to implement Q.
Board representation:  logic, dim-2, polar
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

import types	# for types.SimpleNamespace

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

	# First define what is the state x, and how it is stored.
	# x is composed of (predicate, point) pairs
	# where predicate is just a number from {0...K}
	# size of state = W pairs.
	# output format is the same

	# Each rule is of the form:
	#	Xx ∧ Xx ∧ Xx → Xx
	# number of variables = I = 3

	def __init__(self, input_dim, action_dim, hidden_size, activation=F.relu, init_w=3e-3):
		super(AlgelogicNetwork, self).__init__()

		self.M = 16	# number of rules
		self.J = 3	# number of atoms per rule
		self.I = 3	# number of variables per rule
		self.L = 2	# length of each atom = number of vars or consts
		self.W = 9	# number of atoms (propositions) in Working Memory

		# **** Define M rules
		# Each rule tail consists of 1 atom, with I substitutions
		# Each rule head consists of J atoms, with I substitutions
		# Each atom in a rule has I constants 'c' and γ's
		self.rules = nn.ModuleList()
		for m in range(0, self.M):
			rule = nn.Module()
			rule.tail = nn.Linear(self.I, self.L)
			rule.head = nn.ModuleList()
			for j in range(0, self.J):			# per atom in rule head
				rule.head.append(nn.Linear(self.L, self.I))
			# create constant symbols
			cs = torch.FloatTensor(self.J + 1, self.L).uniform_(-1,1)
			rule.constants = nn.Parameter(cs)
			# create γ values
			γs = torch.FloatTensor(self.J + 1, self.L).uniform_(0,1)
			rule.γs = nn.Parameter(γs)
			self.rules.append(rule)

		self.activation = F.relu

	# **** adjust probability p with weight γ,
	# such that if γ = 0 or close to 0, output p = 1
	#			if γ = 1 or close to 1, output p = p
	# Formula: output = p*t + 1*(1-t)	<-- this is the 'homotopy' trick
	# where t = sigmoid(γ), specifically, t = 1/(1 + exp(-c*(γ - 0.5)))
	# where c = scaling factor to make the sigmoid more steep
	# and the sigmoid is shifted to where the midpoint occurs at γ = 1/2
	def selector(p, γ):
		t = 1.0/(1.0 + exp(-50*(γ - 0.5)))
		return p*t + 1.0 - t

	def softmax(x):
		β = 5		# temperature parameter
		maxes = torch.max(x, 0, keepdim=True)[0]
		x_exp = torch.exp(β * (x - maxes))
		x_exp_sum = torch.sum(x_exp, 0, keepdim=True)
		probs = x_exp / x_exp_sum
		return probs

	def sigmoid(γ):
		steepness = 10.0
		t = 1.0/(1.0 + torch.exp(-steepness*(γ - 0.5)))
		return t

	# **** This is the "matching" or equality function
	# in general, x & y are two vectors
	# in TicTacToe, x & y would be two real numbers ∈ [0,1]
	def match(γ, x, y):
		match_degree = AlgelogicNetwork.sigmoid(γ) * (x - y)**2
		return match_degree

	# **** Algorithm ****
	# for each rule:
	#	for each atom in rule:
	#		try to match WM atoms (which always succeeds to a degree)
	#		for each match
	#			do substitutions (which are cumulative)
	#	output atom (per rule)
	# TV of the conclusion is equal to that of the premises
	# if TV is too low, can that conclusion be disgarded?
	def forward(self, state):
		# TVs of all output predicates:
		P = torch.zeros([self.M], dtype=torch.float)
		state=state.reshape(-1,9,2)
		print('state=', state)

		for m in range(0, self.M):			# for each rule
			rule = self.rules[m]
			for j in range(0, self.J):		# for each atom in rule
				# 1. Calculate matching degrees
				# tv = torch.zeros(self.W, self.I)
				tv = AlgelogicNetwork.match(
					1 - rule.γs[j+1],
					rule.constants[j],
					state )

				""" Below is same operation as above,
					but iterated with loops, for initial testing:
				for w in range(0, self.W):	# for each atom in WM
					# match( rule atom, WM atom )
					tv[w] = AlgelogicNetwork.match(
						1 - rule.γs[j+1],
						rule.constants[j],
						state[0, w] )

					for i in range(0, self.I):
						tv[w,i] = AlgelogicNetwork.match(
							1 - rule.γs[j+1][i],
							rule.constants[j][i],
							state[0, w * self.I + i] )"""

				print("TV=", tv.shape, tv)

				# 2. Do substitutions:
				# copy from state (Working Memory) into variable slots Xs:
				weights = rule.head[j].weight
				print("weights=", weights)
				logits = AlgelogicNetwork.softmax(weights)
				print("logits=", logits.shape, logits)
				print("state=", state.shape, state)
				Xs = torch.matmul(state, logits.mT)
				print("Xs =", Xs.shape, Xs)
				# copy from Xs into OUTPUT proposition:
				weights = rule.tail.weight
				# print("weights=", weights)
				logits = AlgelogicNetwork.softmax(weights)
				print("logits=", logits.shape, logits)
				Ys = torch.matmul(Xs, logits.mT)
				print("Ys =", Ys)
				exit(0)
				# exp to get probability distro over all M conclusions
				# return prob distro for all M conclusions

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
		self.anet = AlgelogicNetwork(state_dim, action_dim, hidden_dim, activation=F.relu).to(device)

		self.q_criterion = nn.MSELoss()
		self.q_optimizer = optim.Adam(self.anet.parameters(), lr=self.lr)

	def choose_action(self, state, deterministic=True):
		state = torch.FloatTensor(state).unsqueeze(0).to(device)
		logits = self.anet(state)
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

		logits = self.anet(state)
		next_logits = self.anet(next_state)

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
