# Just copied some functions from TTT_symmetry.py for easy Python console debugging

def show_vec(board):
	for i in [0, 3, 6]:
		for j in range(3):
			x = board[i + j]
			if x == -1:
				c = 'X'
			elif x == 1:
				c = 'O'
			else:
				c = '-'
			print(c, end='')
	print(end='\n')

def show(board, a=None):
	if type(board).__name__ == 'int':
		board = base3toVec(board)
	for i in [0, 3, 6]:
		for j in range(3):
			x = board[i + j]
			if a == i + j:
				print("\033[42m", end='')
			if x == -1:
				c = '❌'
			elif x == 1:
				c = '⭕'
			else:
				c = '  '
			print(c, end='\033[0m')
		print(end='\n')

# Convert board vector to base-3 number
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

# Convert base-3 number to board vector
def base3toVec(s):
	board = base3toBoard(s)
	board = (9 - len(board)) * [-1] + board
	return board

def base3toBoard(s):
	if s <= 2:
		return [s - 1]
	else:
		return base3toBoard(s // 3) + [(s % 3) - 1]

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

# This function is player-independent
def possible_moves(board):
	moves = []
	for i in range(9):
		if board[i] == 0:
			moves.append(i)
	return moves

# Symmetry of a square, the dihedral group Dih_4
# https://en.m.wikipedia.org/wiki/Examples_of_groups#The_symmetry_group_of_a_square_-_dihedral_group_of_order_8
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

# Apply a group action to a board
def applySym(board, sym):
	newBoard = []
	for j in range(0,9):
		# print(group[sym][j])
		newBoard += [board[group[sym][j]]]
	return newBoard

# **** Find the pair (s,a) in the list eqPairs, ie Q-table entry
def findEntry(s, a):
	# print("pair.shape=",pair.shape)
	j = -1
	for (i, cls) in enumerate(eqPairs):
		if (s,a) in cls:
			j = i
			break
	assert j != -1, "board state " + str((s,a)) + " not found in equivalence classes"
	return j

# **** Same as above, but find all 8 entries of (s,_)
def findEntries(s):
	entries = [float('nan')] * 9
	for i, cls in enumerate(eqPairs):
		for pair in cls:
			if pair[0] == s:
				entries[pair[1]] = i
	assert len(entries) == 9, "state " + str(s) + " has " + str(len(entries)) + " actions instead of 9"
	return entries
