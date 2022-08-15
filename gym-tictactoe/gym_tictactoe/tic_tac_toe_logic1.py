import gym
import numpy
from gym import spaces, error
import xml.etree.ElementTree as ET
import os

# This is a stub for tic-tac-toe logic v1.0
# The code is same as v0, to be modified.

class TicTacToeEnv(gym.Env):

	def __init__(self, symbols, board_size=3, win_size=3):
		super(TicTacToeEnv, self).__init__()
		self.win_size = win_size
		self.board_size = board_size
		self.symbols = {
			symbols[0]: "❌",
			symbols[1]: "⭕",
			"Bad": "!!"
		}
		self.action_space = spaces.Discrete(self.board_size * self.board_size)

		# State space has 9 elements, each element is a vector of dim 3
		self.state_space = spaces.Box(
		# The entries indicate the min and max values of the "box":
		numpy.array(numpy.float32( [0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1] )), \
		numpy.array(numpy.float32( [3,3,1,  3,3,1,  3,3,1,  3,3,1,  3,3,1,  3,3,1,  3,3,1,  3,3,1,  3,3,1] )))

		self.rewards = {
			'still_in_game': 0.0,
			'draw': 10.0,
			'win': 20.0,
			'bad_position': -30.0
			}

	def reset(self):
		self.state_vector = (3 * self.board_size * self.board_size) * [0]
		self.index = 0			# current state_vector position to write into
		self.board = (self.board_size * self.board_size) * [0]
		return numpy.array(self.state_vector)

	# --------------------------------- GAME STATE CHECK -------------------------------------
	def is_win(self):
		if self.check_horizontal():
			return True

		if self.check_vertical():
			return True

		return self.check_diagonal()

	def check_horizontal(self):
		grid = self.board
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
		grid = self.board
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
		grid = self.board
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
			if self.board[i] == 0:
				return False
		return True

	# --------------------------------------- ACTIONS -------------------------------------
	def step(self, action, symbol):
		is_position_already_used = False

		if self.board[action] != 0:
			is_position_already_used = True

		if is_position_already_used:
			self.board[action] = "Bad"
			self.state_vector[self.index] = 0
			self.state_vector[self.index + 1] = 0
			self.state_vector[self.index + 2] = "Bad"
			reward_type = 'bad_position'
			done = True
		else:
			self.board[action] = symbol
			self.state_vector[self.index] = action % 3
			self.state_vector[self.index + 1] = numpy.floor_divide(action, 3)
			self.state_vector[self.index + 2] = symbol

			if self.is_win():
				reward_type = 'win'
				done = True
			elif self.is_draw():
				reward_type = 'draw'
				done = True
			else:
				reward_type = 'still_in_game'
				done = False

		self.index += 3

		return numpy.array(self.state_vector), self.rewards[reward_type], done, {'already_used_position': is_position_already_used}

	# -------------------------------------- DISPLAY ----------------------------------------
	def get_state_vector_to_display(self):
		grid = []
		for value in self.board:
			if value == 0:
				grid.append(0)
			else:
				grid.append(self.symbols[value])
		return grid

	def print_grid_line(self, grid, offset=0):
		print(" " + "-" * (self.board_size * 5 + 1))
		for i in range(self.board_size):
			if grid[i + offset] == 0:
				print(" | " + "  ", end='')
			else:
				print(" | " + str(grid[i + offset]), end='')
		print(" |")

	def display_grid(self, grid):
		print()
		for i in range(0, self.board_size * self.board_size, self.board_size):
			self.print_grid_line(grid, i)

		print(" " + "-" * (self.board_size * 5 + 1))

	def render(self, mode=None, close=False):
		self.display_grid(self.get_state_vector_to_display())

	def close(self):
		return None

	def seed(self, seed=None):
		self.action_space.np_random.seed(seed)
		return [seed]
