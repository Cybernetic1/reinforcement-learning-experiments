import math     # for math.inf = infinity

print("Find all unique board positions of TicTacToe, quotient symmetry")
print("You may modify the initial board position in the code,")
print("or specify a string argument like this: '-1 1 0 1 1 -1 -1 0 0'")
print("where X=-1, O=1, empty=0. Counting from upper left to lower right")

# Empty board
test_board = 9 * [0]

import sys

if len(sys.argv) > 1:
	xs = sys.argv[1].split(' ')
	for i in range(9):
		test_board[i] = int(xs[i])
else:
	# Pre-moves, if any are desired:
	# X|O|
	# O|O|X
	# X| |
	test_board[0] = -1
	test_board[3] = 1
	test_board[6] = -1
	test_board[4] = 1
	# test_board[5] = -1
	# test_board[1] = 1

def show_board(board):
	for i in [0, 3, 6]:
		for j in range(3):
			x = board[i + j]
			if x == -1:
				c = '❌'
			elif x == 1:
				c = '⭕'
			else:
				c = '  '
			print(c, end='')
		print(end='\n')

if test_board != 9 * [0]:
	print("\nInitial board position:")
	show_board(test_board)

group = {
	'e':   [0,1,2,3,4,5,6,7,8],
	'a':   [6,3,0,7,4,1,8,5,2],
	'a2':  [8,7,6,5,4,3,2,1,0],
	'a3':  [2,5,8,1,4,7,0,3,6],
	'b':   [2,1,0,5,4,3,8,7,6],
	'ab':  [0,3,6,1,4,7,2,5,8],
	'a2b': [6,7,8,3,4,5,0,1,2],
	'a3b': [8,5,2,7,4,1,6,3,0]
}

def applySym(board, sym):
	newBoard = []
	for j in range(0,9):
		# print(group[sym][j])
		newBoard += [board[group[sym][j]]]
	return newBoard

def base3toBoard(s):
	if s <= 2:
		return [s - 1]
	else:
		return base3toBoard(s // 3) + [(s % 3) - 1]

def base3(board):
	s = (((((((					\
		board[0] * 3 + 3 +		\
		board[1]) * 3 + 3 +		\
		board[2]) * 3 + 3 +		\
		board[3]) * 3 + 3 +		\
		board[4]) * 3 + 3 +		\
		board[5]) * 3 + 3 +		\
		board[6]) * 3 + 3 +		\
		board[7]) * 3 + 3 +		\
		board[8]+1
	return s

# Enumerate all board positions = 3^9 = 19683
# This includes many impossible boards, such as all X's.
# But this part is useful for checking correctness of functions.
eqClasses = []
for s in range(0,19683):
	duplicate = False
	for c in eqClasses:
		if s in c:
			duplicate = True
	if duplicate:
		continue

	# convert base-3 number to board vector
	board = base3toBoard(s)
	board = (9 - len(board)) * [-1] + board
	# print(s - base3(board), s, board)
	# print(board)
	# show_board(board)
	print(s)

	# for each board position, find all its symmetric positions
	# put into list of equivalence classes
	cls = set()
	for sym in ['a','a2','a3','b','ab','a2b','a3b']:
		board2 = applySym(board, sym)
		# convert to base-3 number
		s2 = base3(board2)
		cls.add(s2)
		# print('-------------')
		# show_board(board2)

	"""duplicate = False		# This seems never true
	for c in eqClasses:
		for c2 in cls:
			if c2 in c:
				duplicate = True
	if duplicate:
		print("*********************")
		continue"""

	cls.add(s)	
	eqClasses += [cls]
	# print('=====================================\n')

print(eqClasses)
print("Total classes = ", len(eqClasses))

eqClasses = []
# **** Find all symmetries of TTT
def allSyms(board, player):

	if player == -1:
		# **** Find all possible next moves for player 'X'
		moves = possible_moves(board)

		# For each possible board, find its base-3 number
		# check for duplicates in equivalence class list
		# Then find all its equivalent forms, add to eqClasses list
		for m in moves:
			new_board = board.copy()
			new_board[m] = -1		# Player 'X'
			s = base3(new_board)

			duplicate = False
			for c in eqClasses:
				if s in c:
					duplicate = True
			if duplicate:
				continue		# next board

			# If this an ending move?
			r = game_over(new_board, -1)
			if r is not None:
				continue
			else:
				allSyms(new_board, 1)		# next player
		#show_board(board)
		print("X's turn.  Expectation w.r.t. Player X =", max_v, end='\r')
		return max_v

	elif player == 1:
		# **** Find all possible next moves for player 'O'
		moves = possible_moves(board)

		for m in moves:
			new_board = board.copy()
			new_board[m] = 1		# Player 'O'

			# If this an ending move?
			r = game_over(new_board, 1)
			if r is not None:
				if r == 10:				# draw is +10 for either player
					Rx += r * p
				else:
					Rx -= r * p			# sign of reward is reversed
			else:
				allSyms(new_board, -1)
		#show_board(board)
		print("O's turn.  Expectation w.r.t. Player X =", Rx, end='\r')
		return Rx

def possible_moves(board):
	moves = []
	for i in range(9):
		if board[i] == 0:
			moves.append(i)
	return moves

# Check only for the given player.
# Return reward w.r.t. the specific player.
def game_over(board, player):
	# check horizontal
	for i in [0, 3, 6]:     # for each row
		if board[i + 0] == player and \
		   board[i + 1] == player and \
		   board[i + 2] == player:
			return 20

	# check vertical
	for j in [0, 1, 2]:     # for each column
		if board[3 * 0 + j] == player and \
		   board[3 * 1 + j] == player and \
		   board[3 * 2 + j] == player:
			return 20

	# check diagonal
	if board[0 + 0] == player and \
	   board[3 * 1 + 1] == player and \
	   board[3 * 2 + 2] == player:
		return 20

	# check backward diagonal
	if board[3 * 0 + 2] == player and \
	   board[3 * 1 + 1] == player and \
	   board[3 * 2 + 0] == player:
		return 20

	# return None if game still open
	for i in [0, 3, 6]:
		for j in [0, 1, 2]:
			if board[i + j] == 0:
				return None

	# For one version of gym TicTacToe, draw = 10 regardless of player;
	# Another way is to assign draw = 0.
	return 10
