"""
Discrete Q table that does not need deep learning

Using:
gym: 0.8.0
"""

import random
import numpy as np

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
		state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
		# print("sampled state=", state)
		# print("sampled action=", action)
		'''
		the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
		zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
		the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
		np.stack((1,2)) => array([1, 2])
		'''
		return state, action, reward, next_state, done

	def __len__(self):
		return len(self.buffer)

class Qtable(nn.Module):

	def __init__(
			self,
			action_dim,
			state_dim,
			learning_rate = 3e-4,
			gamma = 0.9 ):
		super(SAC, self).__init__()

		self.action_dim = action_dim
		self.state_dim = state_dim

		# dim of Q-table = board-size x action_dim
		self.Qtable = np.zeros(state_dim, action_dim)
		# self.Qtable = [ [0] * action_dim for i in range(state_dim) ]

		self.lr = learning_rate
		self.gamma = gamma

		self.replay_buffer = ReplayBuffer(int(1e6))

	def choose_action(self, state, deterministic=False):
		Q_a = self.Qtable[state, :]
		action = np.argmax(Q_a)
		return action

	def update(self, batch_size, reward_scale, gamma=0.99):
		alpha = 1.0  # trade-off between exploration (max entropy) and exploitation (max Q)

		state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
		# print('sample (state, action, reward, next state, done):', state, action, reward, next_state, done)

		# **** Training Policy Function

		return

	def net_info(self):
		config = "(3^9x9)"
		return (config, 3 ** 10)

	def play_random(self, state, action_space):
		# Select an action (0-9) randomly
		# NOTE: random player never chooses occupied squares
		while True:
			action = action_space.sample()
			occupied = state[action]
			if occupied > -0.1 and occupied < 0.1:
				break
		return action

	def save_net(self, fname):
		print("Model not saved.")

	def load_net(self, fname):
		print("Model not loaded.")
