import math		# for math.inf = infinity

print("Calculate optimal expectation value of TicTacToe")
print("from the perspective of 'X' = first player.")
print("Assume both players perfectly avoid illegal moves.")
print("Player 'X' always chooses the move with maximum expectation value.")
print("Player 'O' always plays all available moves with equal probability.")
print("You may modify the initial board position in the code.")

# Empty board
test_board = 9 * [0]

# Pre-moves, if any are desired:
# X|O|
# O|O|X
# X| |
test_board[0] = -1
test_board[3] = 1
test_board[6] = -1
test_board[4] = 1
test_board[5] = -1
test_board[1] = 1

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

# **** Calculate expectation value of input board position
def expectation(board, player):

	if player == -1:
		# **** Find all possible next moves for player 'X'
		moves = possible_moves(board)

		# Calculate expectation of these moves;
		# Player 'X' will only choose the one of maximum value.
		max_v = - math.inf
		for m in moves:
			new_board = board.copy()
			new_board[m] = -1		# Player 'X'

			# If this an ending move?
			r = game_over(new_board, -1)
			if r is not None:
				if r > max_v:
					max_v = r
			else:
				v = expectation(new_board, 1)
				if v > max_v:
					max_v = v
		# show_board(board)
		print("X's turn.  Expectation w.r.t. Player X =", max_v, end='\r')
		return max_v

	elif player == 1:
		# **** Find all possible next moves for player 'O'
		moves = possible_moves(board)
		# These moves have equal probability
		# print(board, moves)
		p = 1.0 / len(moves)

		# Calculate expectation of these moves;
		# Player 'O' chooses one of them randomly.
		Rx = 0.0		# reward from the perspective of 'X'
		for m in moves:
			new_board = board.copy()
			new_board[m] = 1		# Player 'O'

			# If this an ending move?
			r = game_over(new_board, 1)
			if r is not None:
				if r == 10:				# draw is +10 for either player
					Rx += r * p
				else:
					Rx += - r * p		# sign of reward is reversed
			else:
				v = expectation(new_board, -1)
				Rx += v * p
		# show_board(board)
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
	for i in [0, 3, 6]:		# for each row
		if board[i + 0] == player and board[i + 1] == player and board[i + 2] == player:
			return 20

	# check vertical
	for j in [0, 1, 2]:		# for each column
		if board[3 * 0 + j] == player and board[3 * 1 + j] == player and board[3 * 2 + j] == player:
			return 20

	# check diagonal
	if board[0 + 0] == player and board[3 * 1 + 1] == player and board[3 * 2 + 2] == player:
		return 20

	# check backward diagonal
	if board[3 * 0 + 2] == player and board[3 * 1 + 1] == player and board[3 * 2 + 0] == player:
		return 20

	# return None if game still open
	for i in [0, 3, 6]:
		for j in [0, 1, 2]:
			if board[i + j] == 0:
				return None

	# For one version of gym TicTacToe, draw = 10 regardless of player;
	# Another way is to assign draw = 0.
	return 10

print("\u001b[2K\nOptimal value =", expectation(test_board, -1) )
