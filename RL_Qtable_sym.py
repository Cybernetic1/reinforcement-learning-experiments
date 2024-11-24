"""
Discrete Q table that does not need deep learning, with dihedral symmetry.
This version accounts for symmetry of (s,a) pairs.
There are 4520 non-end states, without symmetry.
Thus 4520 x 9 = 40680 (s,a) pairs, without symmetry.
After finding symmetries, reduces to 5263 equivalence classes.
Thus 5263 is the size of our Q-table.
This is perhaps the most efficient Q-table we can hope for, exploiting all symmetries.
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
		return self.buffer[self.position -1][2]

	def sample(self, batch_size):
		# **** Old method: random sample
		# batch = random.sample(self.buffer, batch_size)
		# New method uses the latest data, seems to converge a bit faster
		# initially, but overall performance is similar to old method
		if self.position >= batch_size:
			batch = self.buffer[self.position - batch_size : self.position]
		else:
			batch = self.buffer[: self.position] + self.buffer[-(batch_size - self.position) :]
		assert len(batch) == batch_size, "batch size incorrect"

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

class Qtable():

	from eqPairs import eqPairs

	# The first state {9841} is the "clean board"; following by ILLEGAL, WIN, LOSE, in that order
	ILLEGAL = 19682
	WIN = 4918
	LOSE = 5029
	DRAW = 4880

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

		# dim of Q-table
		self.Qtable = np.zeros(5263)

		self.replay_buffer = ReplayBuffer(int(1e5))	# originally 1e6

	# convert state-vector into a base-3 number
	def state_num(state):
		if 2 in state:
			return Qtable.ILLEGAL
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

	# **** Find the pair (s,a) in the list eqPairs, ie Q-table entry
	def findEntry(s, a):
		# print("pair.shape=",pair.shape)
		j = -1
		for (i, cls) in enumerate(Qtable.eqPairs):
			if (s,a) in cls:
				j = i
				break
		assert j != -1, "board state " + str((s,a)) + " not found in equivalence classes"
		return j

	# **** Same as above, but find all 8 entries of (s,_)
	def findEntries(s):
		entries = []
		for (i,cls) in enumerate(Qtable.eqPairs):
			for pair in cls:
				if pair[0] == s:
					entries += [i]
		assert len(entries) == 9, "state " + str(s) + " has " + str(len(entries)) + " actions instead of 9"
		return entries

	def choose_action(self, state, deterministic=False):
		s = Qtable.state_num(state)
		logits = []			# logits = Q(s,a)
		# collect all 8 Q-table entries that match the state s
		for j, cls in enumerate(Qtable.eqPairs):
			for pair in cls:
				if pair[0] == s:
					logits += [self.Qtable[j]]
		assert len(logits) == 9, "State " + str(s) + " has " + str(len(logits)) + " actions instead of 8"
		logits = np.array(logits)
		probs   = np.exp(logits) / np.exp(logits).sum(axis=0)	# softmax
		# print("logits, probs =", logits, probs)
		action  = np.random.choice([0,1,2,3,4,5,6,7,8], 1, p=probs)[0]
		# action = np.argmax(logits)		# deterministic
		# print("chosen action=", action)
		return action

	def show_board(board):
		for i in [0, 3, 6]:
			for j in range(3):
				x = board[i + j]
				if x == -1:
					c = '❌'
				elif x == 1:
					c = '⭕'
				elif x == 2:
					c = '🟨'
				else:
					c = '  '
				print(c, end='')
			print(end='\n')

	def update(self, batch_size, reward_scale, gamma=0.99):
		alpha = 1.0  # trade-off between exploration (max entropy) and exploitation (max Q)

		states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
		# print('sample (state, action, reward, next state, done):', states, actions, rewards, next_states, dones)

		# convert state-vector to a base-3 number
		s = (((((((					\
			states[:,0] * 3 + 3 +	\
			states[:,1]) * 3 + 3 +	\
			states[:,2]) * 3 + 3 +	\
			states[:,3]) * 3 + 3 +	\
			states[:,4]) * 3 + 3 +	\
			states[:,5]) * 3 + 3 +	\
			states[:,6]) * 3 + 3 +	\
			states[:,7]) * 3 + 3 +	\
			states[:,8] + 1
		applyall = np.vectorize(Qtable.findEntry)
		j = applyall(s, actions)

		# for st in next_states:
		#	Qtable.show_board(st)
		#	print('---------------')
		# print("next states =", next_states.shape, next_states)
		k = np.array(list(map(Qtable.findEntries, list(map(Qtable.state_num, next_states)))))

		# **** Train Q function, this is just Bellman equation:
		# Q(st,at) += η [ R + γ max_a Q(s_t+1,a) - Q(st,at) ]
		self.Qtable[j] += self.lr *( rewards + self.gamma * np.max(self.Qtable[k, :]) - self.Qtable[j] )
		return

	def visualize_q(self, board):
		# convert board vector to a base-3 number
		s = (((((((					\
			board[0] * 3 + 3 +	\
			board[1]) * 3 + 3 +	\
			board[2]) * 3 + 3 +	\
			board[3]) * 3 + 3 +	\
			board[4]) * 3 + 3 +	\
			board[5]) * 3 + 3 +	\
			board[6]) * 3 + 3 +	\
			board[7]) * 3 + 3 +	\
			board[8] + 1
		j = Qtable.findClass(s)
		logits = self.Qtable[j, :]
		probs  = np.exp(logits) / np.exp(logits).sum(axis=0)	# softmax
		return probs

	def net_info(self):
		config = "(5263)"
		return (config, 5263)

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
		np.save(fname, self.Qtable)
		print("Q-table saved.")

	def load_net(self, fname):
		self.Qtable = np.load(fname)
		print("Q-table loaded.")
