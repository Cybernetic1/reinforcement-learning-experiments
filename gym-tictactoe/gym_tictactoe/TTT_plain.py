# action space = { 0 ... 8 }
# state space = board vector = {-1, 0, 1} ^ 9

import gym
import numpy
from gym import spaces, error
import xml.etree.ElementTree as ET
import os

import websockets
from websockets.sync.client import connect
import json

class TicTacToeEnv(gym.Env):

	def __init__(self, symbols=[-1,1], board_size=3, win_size=3):
		self.win_size = win_size
		self.board_size = board_size
		self.symbols = {
			symbols[0]: "X",
			symbols[1]: "O",
			2: "!"
			}
		self.action_space = spaces.Discrete(self.board_size * self.board_size)

		self.state_space = spaces.Box( \
			numpy.float32(numpy.array([-1,-1,-1,-1,-1,-1,-1,-1,-1])), \
			numpy.float32(numpy.array([+1,+1,+1,+1,+1,+1,+1,+1,+1])) )
		
		# Gym 0.26+ requires observation_space to be defined before super().__init__()
		self.observation_space = self.state_space
		
		super(TicTacToeEnv, self).__init__()

		self.rewards = {
			'in_game': 0.0,
			'draw': 10.0,
			'win': 20.0,
			'bad': -30.0
			}

	def reset(self, seed=None, options=None):
		if seed is not None:
			self.action_space.seed(seed)
		self.state_vector = (self.board_size * self.board_size) * [0]
		return numpy.array(self.state_vector), {}

	# --------------------------------- GAME STATE CHECK -------------------------------------
	def is_win(self):
		if self.check_horizontal():
			return True

		if self.check_vertical():
			return True

		return self.check_diagonal()

	def check_horizontal(self):				# returns True if EITHER player has a full row
		grid = self.state_vector
		cnt = 0
		for i in range(0, self.board_size * self.board_size, self.board_size):
			cnt = 0
			k = i
			for j in range(1, self.board_size):
				(cnt, k) = (cnt + 1, k) if (grid[k] == grid[i + j] and grid[k] != 0) else (0, i + j)
				if cnt == self.win_size - 1:
					return True

		return False

	def check_vertical(self):
		grid = self.state_vector
		cnt = 0
		for i in range(0, self.board_size):
			cnt = 0
			k = i
			for j in range(self.board_size, self.board_size * self.board_size, self.board_size):
				(cnt, k) = (cnt + 1, k) if (grid[k] == grid[i + j] and grid[k] != 0) else (0, i + j)
				if cnt == self.win_size - 1:
					return True

		return False

	def check_diagonal(self):
		grid = self.state_vector
		m = self.to_matrix(grid)
		m = numpy.array(m)

		for i in range(self.board_size - self.win_size + 1):
			for j in range(self.board_size - self.win_size + 1):
				sub_matrix = m[i:self.win_size + i, j:self.win_size + j]

				if self.check_matrix(sub_matrix):
					return True

		return False

	def to_matrix(self, grid):
		m = []
		for i in range(0, self.board_size * self.board_size, self.board_size):
			m.append(grid[i:i + self.board_size])
		return m

	def check_matrix(self, m):
		cnt_primary_diag = 0
		cnt_secondary_diag = 0
		for i in range(self.win_size):
			for j in range(self.win_size):
				if i == j and m[0][0] == m[i][j] and m[0][0] != 0:
					cnt_primary_diag += 1

				if i + j == self.win_size - 1 and m[0][self.win_size - 1] == m[i][j] and m[0][self.win_size - 1] != 0:
					cnt_secondary_diag += 1

		return cnt_primary_diag == self.win_size or cnt_secondary_diag == self.win_size

	def is_draw(self):
		for i in range(self.board_size * self.board_size):
			if self.state_vector[i] == 0:
				return False
		return True

	# --------------------------------------- ACTIONS -------------------------------------
	def step(self, action, symbol):
		# print(symbol, ":", type(symbol))
		is_position_already_used = False

		if self.state_vector[action] != 0:
			is_position_already_used = True

		if is_position_already_used:
			self.state_vector[action] = 2		# signifies "bad"
			reward_type = 'bad'
			done = True
		else:
			self.state_vector[action] = symbol

			if self.is_win():
				reward_type = 'win'
				done = True
			elif self.is_draw():
				reward_type = 'draw'
				done = True
			else:
				reward_type = 'in_game'
				done = False

		return numpy.array(self.state_vector), self.rewards[reward_type], done, False, {'reward_type': reward_type}

	# -------------------------------------- DISPLAY ----------------------------------------
	def get_state_vector_to_display(self):
		new_state_vector = []
		for value in self.state_vector:
			if value == 0:
				new_state_vector.append(value)
			else:
				new_state_vector.append(self.symbols[value])
		return new_state_vector

	def print_grid_line(self, grid, offset=0):
		print(" " + "-" * (self.board_size * 4 + 1))
		for i in range(self.board_size):
			if grid[i + offset] == 0:
				print(" | " + " ", end='')
			else:
				print(" | " + str(grid[i + offset]), end='')
		print(" |")

	def display_grid(self, grid):
		for i in range(0, self.board_size * self.board_size, self.board_size):
			self.print_grid_line(grid, i)

		print(" " + "-" * (self.board_size * 4 + 1))
		print()

	def render(self, mode=None):
		if mode == 'HTML':
			with connect("ws://localhost:5678") as websocket:
				websocket.send(json.dumps(self.state_vector))
		else:
			self.display_grid(self.get_state_vector_to_display())

	def close(self):
		return None

	def seed(self, seed=None):
		if seed is not None:
			self.action_space.seed(seed)
		return [seed]
