print("Find all unique board positions of TicTacToe, quotient symmetry\n")

# TO-DO:
# * why non-equivalent (s,a) pairs are mixed up

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

ILLEGAL = 19682
WIN = 4918
LOSE = 5029
DRAW = 19682
eqClasses = [set([9841]), set([ILLEGAL]), set([WIN]), set([LOSE]), set([DRAW])]

# Add all other equivalent members to a class
def addEqvs(board, cls):
	for sym in ['a','a2','a3','b','ab','a2b','a3b']:
		board2 = applySym(board, sym)
		# convert to base-3 number
		s2 = base3(board2)
		cls.add(s2)

steps = 0		# number of states traversed, with repeats
end_steps = 0	# number of end states reached, with repeats
reachables = set()
ends = set()

# **** Find all reachable states of TicTacToe
def play_all(board, player):
	global reachables, ends
	global steps, end_steps

	reachables.add(base3(board))
	steps += 1
	# **** Find all possible next moves for player 'X' or 'O'
	moves = possible_moves(board)

	# For each possible board, find its base-3 number
	# check for duplicates in equivalence class list
	# Then find all its equivalent forms, add new equivalence class
	for m in moves:
		new_board = board.copy()
		new_board[m] = player

		# If this an ending move? If yes, terminate recursion
		r = game_over(new_board, player)
		if r is not None:	# game-over'd
			ends.add(base3(new_board))
			end_steps += 1
			continue		# next move, no need to recurse
		play_all(new_board, -player)		# recurse with next player

print("Without symmetry...")
play_all([0]*9, -1)
print("Total # of non-end moves played =", steps)
print("Total # of games played =", end_steps)
count = len(reachables)
count_end = len(ends)
print("Total # of non-end states =", count)
print("Total # of end states =", count_end)
print("Total # of reachable states =", count + count_end)

ans = input("\nWrite all reachable states to reachables.py? [y/N]")
if ans == 'Y' or ans == 'y':
	f1 = open("reachables.py", 'w')
	f1.write("reachables =")
	f1.write(str(reachables))
	f1.close()
exit(0)

def allSyms(states):
	eqClasses = []
	for s in states:

		# check if duplicated
		duplicate = False
		for cls in eqClasses:
			if s in cls:
				duplicate = True
		if duplicate:
			continue		# next state

		# find all s's symmetries and add to new class
		cls = set()
		cls.add(s)
		board = base3toVec(s)
		addEqvs(board, cls)
		eqClasses += [cls]

	return eqClasses

print("\nWith symmetry...")
eqStates = allSyms(reachables)
num_states = len(eqStates)
print("Total # of non-end states =", num_states)
eqEndStates = allSyms(ends)
num_ends = len(eqEndStates)
print("Total # of end states =", num_ends)
print("Total # of reachable states =", num_states + num_ends)

reachablePairs = [(s,a) for s in reachables for a in [0,1,2,3,4,5,6,7,8]]
print("\nTotal # of (s,a) pairs (without symmetry) =", len(reachablePairs))

# **** Find all equivalence classes of (s,a) pairs
# code is same as the previous except actions are also transformed
def allSymPairs(pairs):
	eqClasses = []
	for (s,a) in pairs:

		# check if duplicated
		duplicate = False
		for cls in eqClasses:
			if (s,a) in cls:
				duplicate = True
		if duplicate:
			continue		# next state

		# find all s's symmetries and add to new class
		cls = set()
		cls.add((s,a))

		board = base3toVec(s)
		for sym in ['a','a2','a3','b','ab','a2b','a3b']:
			board2 = applySym(board, sym)
			s2 = base3(board2)			# convert to base-3 number
			a2 = group[sym].index(a)	# apply transform to action as well
			cls.add((s2,a2))
		eqClasses += [cls]

	return eqClasses

eqPairs = allSymPairs(reachablePairs)
print("Total # of (s,a) pairs (with symmetry) =", len(eqPairs))

# New method is to build dictionary (s,a) --> class number
Qdict = {}
for i, cls in enumerate(eqPairs):
	for (q,a) in cls:
		Qdict[(q,a)] = i
print("Qdict =", Qdict)

exit(0)

ans = input("\nWrite output to eqPairs.py? [y/N]")
if ans == 'Y' or ans == 'y':
	f1 = open("eqPairs.py", 'w')
	f1.write("eqPairs =")
	f1.write(str(eqPairs))
	f1.close()

exit(0)
count2 = 0
count_end2 = 0

# **** Find all symmetries of TTT, by recursively playing game
def allSyms2(board, player):
	global eqClasses
	global count2, count_end2
	# **** Find all possible next moves for player 'X' or 'O'
	moves = possible_moves(board)

	# For each possible board, find its base-3 number
	# check for duplicates in equivalence class list
	# Then find all its equivalent forms, add new equivalence class
	for m in moves:
		new_board = board.copy()
		new_board[m] = player
		s = base3(new_board)
		count2 += 1

		duplicate = False
		for c in eqClasses:
			if s in c:
				duplicate = True
		if duplicate:
			continue		# next move

		# If this an ending move? If yes, terminate recursion
		r = game_over(new_board, player)
		if r is not None:	# game-over'd
			count_end2 += 1
			# add to WIN or LOSE classes, respectively
			if r == 10:				# DRAW
				eqClasses[4].add(s)
				addEqvs(new_board, eqClasses[4])
			elif player == -1:		# WIN
				eqClasses[2].add(s)
				addEqvs(new_board, eqClasses[2])
			else:					# LOSE
				eqClasses[3].add(s)
				addEqvs(new_board, eqClasses[3])
			continue		# next move, no need to recurse

		# add new class of s and its equivalents
		cls = set()
		cls.add(s)
		addEqvs(new_board, cls)
		eqClasses += [cls]

		allSyms(new_board, -player)		# recurse with next player

print("\nWith symmetry...")
allSyms([0] * 9, -1)
print("Total number of reachable states (including end states) =", count2)
print("Total number of end states =", count_end2)
print("Restricted total classes =", len(eqClasses))

ans = input("\nWrite output to eqClasses.py? [y/N]")
if ans == 'Y' or ans == 'y':
	f1 = open("eqClasses.py", 'w')
	f1.write("eqClasses =")
	f1.write(str(eqClasses))
	f1.close()

# *************** This is an older incorrect version ******************

# Enumerates all board positions = 3^9 = 19683
# This includes many impossible boards, such as all X's.
# But this part is useful for checking correctness of functions.
def allSyms_incorrect():
	eqClasses = []
	for s in range(0,19683):
		duplicate = False
		for c in eqClasses:
			if s in c:
				duplicate = True
		if duplicate:
			continue

		# convert base-3 number to board vector
		board = base3toVec(s)
		# print(s - base3(board), s, board)
		# print(board)
		# show_board(board)
		print(s, end='\t')

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
