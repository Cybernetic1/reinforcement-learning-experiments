"""
Deep Q Network
modified DQN where state space consists of logic propositions
and each proposition is a 1-dim vector (ie, just a number)

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
		return self.buffer[-1][2]

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

class QNetwork(nn.Module):
	def __init__(self, input_dim, action_dim, hidden_size, activation=F.relu, init_w=3e-3):
		super(QNetwork, self).__init__()

		self.linear1 = nn.Linear(input_dim, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, hidden_size)
		self.linear4 = nn.Linear(hidden_size, hidden_size)

		self.logits_linear = nn.Linear(hidden_size, action_dim)
		self.logits_linear.weight.data.uniform_(-init_w, init_w)
		self.logits_linear.bias.data.uniform_(-init_w, init_w)

		self.activation = F.relu

	def forward(self, state):
		x = self.activation(self.linear1(state))
		x = self.activation(self.linear2(x))
		x = self.activation(self.linear3(x))
		x = self.activation(self.linear4(x))

		logits = self.logits_linear(x)
		# logits = F.leaky_relu(self.logits_linear(x))
		return logits

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

		hidden_dim = 32
		self.q_net = QNetwork(state_dim, action_dim, hidden_dim, activation=F.relu).to(device)

		self.q_criterion = nn.MSELoss()
		self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

	def choose_action(self, state, deterministic=True):
		state = torch.FloatTensor(state).unsqueeze(0).to(device)

		logits = self.q_net(state)
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

		logits = self.q_net(state)
		next_logits = self.q_net(next_state)

		q = logits[range(logits.shape[0]), action]
		m = torch.max(next_logits, 1, keepdim=False).values
		target_q = torch.where(done, reward, reward + self.gamma * m)
		q_loss = self.q_criterion(q, target_q.detach())

		self.q_optimizer.zero_grad()
		q_loss.backward()
		self.q_optimizer.step()

		return

	def net_info(self):
		config = "(9)-32-32-32-32-32-(9)"
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
		# NOTE: random player never chooses occupied squares
		empties = [1,2,3,4,5,6,7,8,9]
		# Find and collect all empty squares
		# scan through all 9 propositions, each proposition is a 1-vector
		for i in range(0, 9):
			# 'proposition' is just 1 number
			proposition = state[i]
			if proposition > 0:
				empties.remove(proposition)
			elif proposition < 0:
				empties.remove(-proposition)
		# Select an available square randomly
		action = random.sample(empties, 1)[0] - 1
		return action

	def save_net(self, fname):
		print("Model not saved.")

	def load_net(self, fname):
		print("Model not loaded.")
