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
		return self.buffer[self.position -1 ][2]

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

class Qtable():

	def __init__(
			self,
			action_dim,
			state_dim,
			learning_rate = 3e-4,
			gamma = 0.9 ):
		super(Qtable, self).__init__()

		self.action_dim = action_dim
		self.state_dim = state_dim
		self.lr = learning_rate
		self.gamma = gamma

		# dim of Q-table = 3^9 x 9
		self.Qtable = np.zeros((3 ** state_dim, action_dim))

		self.replay_buffer = ReplayBuffer(int(1e6))

	# convert state-vector into a base-3 number
	def state_num(self, state):
		s = (((((((					\
			state[0] * 3 + 3 +		\
			state[1]) * 3 + 3 +		\
			state[2]) * 3 + 3 +		\
			state[3]) * 3 + 3 +		\
			state[4]) * 3 + 3 +		\
			state[5]) * 3 + 3 +		\
			state[6]) * 3 + 3 +		\
			state[7]) * 3 + 3 +		\
			state[8]+1
		return s

	def choose_action(self, state, deterministic=False):
		s = self.state_num(state)
		logits  = self.Qtable[s, :] 		# = Q(s,a)
		probs   = np.exp(logits) / np.exp(logits).sum(axis=0)	# softmax
		# print("logits, probs =", logits, probs)
		action  = np.random.choice([0,1,2,3,4,5,6,7,8], 1, p=probs)[0]
		# action = np.argmax(logits)		# deterministic
		# print("chosen action=", action)
		return action

	def update(self, batch_size, reward_scale, gamma=0.99):
		alpha = 1.0  # trade-off between exploration (max entropy) and exploitation (max Q)

		state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
		# print('sample (state, action, reward, next state, done):', state, action, reward, next_state, done)

		# convert state-vector to a base-3 number
		s = (((((((					\
			state[:,0] * 3 + 3 +	\
			state[:,1]) * 3 + 3 +	\
			state[:,2]) * 3 + 3 +	\
			state[:,3]) * 3 + 3 +	\
			state[:,4]) * 3 + 3 +	\
			state[:,5]) * 3 + 3 +	\
			state[:,6]) * 3 + 3 +	\
			state[:,7]) * 3 + 3 +	\
			state[:,8] + 1
		# print("s=", s)

		# **** Train Q function, this is just Bellman equation:
		# Q(st,at) += η [ R + γ max_a Q(s_t+1,a) - Q(st,at) ]
		self.Qtable[s, action] += self.lr *( reward + self.gamma * np.max(self.Qtable[next_state, :]) - self.Qtable[s, action] )
		return

	def net_info(self):
		config = "(3^9x9)"
		return (config, 3 ** 10)

	def play_random(self, state, action_space):
		# Select an action (0-9) randomly
		# NOTE: random player never chooses occupied squares
		empties = [0,1,2,3,4,5,6,7,8]
		# Find and collect all empty squares
		# scan through board vector
		for i in range(0, 9):
			# 'proposition' is a numpy array[3]
			if state[i] == 1 or state[i] == -1:
				empties.remove(i)
		# Select an available square randomly
		action = random.sample(empties, 1)[0]
		return action

	def save_net(self, fname):
		print("Model not saved.")

	def load_net(self, fname):
		print("Model not loaded.")
