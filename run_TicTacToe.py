"""
Reinforcement Learning, Tic Tac Toe experiments

With a choice of engines:
- PyTorch: 1.9.0+cpu
- TensorFlow: 2.0

With a choice of representations:
- board vector
- logic propositions

With a choice of algorithms:
- fully-connected
- symmetric NN
- Transformer
"""
print("    Engine\tAlgo\tStructure\t\tRepresentation")
print("=" * 70)
print("0.  Python\tQ-table\tno NN\t\t\tboard vector")
print("1.  Python\tQ-table\tno NN\t\t\tboard vector with symmetry")
print("10. PyTorch\tPG\tsymmetric NN\t\tlogic, dim3")
print("11. PyTorch\tPG\tfully-connected\t\tboard vector")
print("12. TensorFlow\tPG\tsymmetric NN\t\tlogic, dim3")
print("13. TensorFlow\tPG\tfully-connected\t\tboard vector")
print("14. PyTorch\tPG\tTransformer\t\tlogic, dim3")
print("15. PyTorch\tSAC\tfully-connected\t\tboard vector\n")

print("20. PyTorch\tDQN\tfully-connected\t\tboard vector")
print("21. PyTorch\tDQN\tTransformer\t\tlogic, dim3")
print("22. PyTorch\tDQN\tfully-connected\t\tlogic, dim3")
print("23. PyTorch\tDQN\tfully-connected\t\tlogic, dim1")
print("24. PyTorch\tDQN\tsymmetric NN\t\tlogic, dim3")
print("25. PyTorch\tDQN\tloop symNN\t\tlogic, dim2")
print("26. PyTorch\tDQN\tshrink until fail\tlogic, dim2")
print("27. PyTorch\tDQN\tshrink symNN\t\tlogic, dim2")
print("28. PyTorch\tDQN\tlogic-with-vars\t\tlogic, dim2 polar")
print("29. PyTorch\tDQN\trelation-graph\t\tlogic, dim2 polar")
config = int(input("Choose config: ") or '0')

import gym

if config == 0:
	from RL_Qtable import Qtable
	tag = "Qtable"
elif config == 1:
	from RL_Qtable_sym import Qtable
	tag = "Qtable.sym"
elif config == 10:
	from PG_symNN_pyTorch import PolicyGradient
	tag = "symNN.pyTorch"
elif config == 11:
	from PG_full_pyTorch import PolicyGradient
	tag = "full.pyTorch"
elif config == 12:
	from PG_symNN_TensorFlow import PolicyGradient
	tag = "symNN.TensorFlow"
elif config == 13:
	from PG_full_TensorFlow import PolicyGradient
	tag = "full.TensorFlow"
elif config == 14:
	from PG_Transformer_pyTorch import PolicyGradient
	tag = "Transformer.pyTorch"
elif config == 15:
	from SAC_full_pyTorch import SAC, ReplayBuffer
	tag = "SAC.full.pyTorch"

elif config == 20:
	from RL_DQN import DQN
	tag = "DQN.BoardVec"
elif config == 21:
	from DQN_Transformer import DQN, ReplayBuffer
	tag = "DQN.Transformer.pyTorch"
elif config == 22:
	from DQN_full import DQN, ReplayBuffer
	tag = "DQN.full"
elif config == 23:
	from DQN_full_dim1 import DQN, ReplayBuffer
	tag = "DQN.full-1D"
elif config == 24:
	from DQN_symNN import DQN, ReplayBuffer
	tag = "DQN.symNN"
elif config == 25:
	from DQN_loop import DQN, ReplayBuffer
	tag = "DQN.loop"
elif config == 26:
	from DQN_shrink import DQN, ReplayBuffer
	tag = "DQN.shrink"
elif config == 27:
	from DQN_shrink_SymNN import DQN, ReplayBuffer
	tag = "DQN.shrink-SymNN"
elif config == 28:
	from DQN_logic_with_vars import DQN, ReplayBuffer
	tag = "DQN.logic_with_vars"
elif config == 29:
	from DQN_relation_graph import DQN, ReplayBuffer
	tag = "DQN.relation_graph"

import sys
sys.path.insert(0, './gym-tictactoe')
import gym_tictactoe
if config in [10, 12, 14, 21, 22, 24]:
	env = gym.make('TicTacToe-logic-v0', symbols=[-1, 1], board_size=3, win_size=3)
elif config == 23:
	env = gym.make('TicTacToe-logic-dim1-v0', symbols=[-1, 1], board_size=3, win_size=3)
elif config == 25:
	env = gym.make('TicTacToe-logic-dim2-intermediate-v0', symbols=[-1, 1], board_size=3, win_size=3)
elif config in [26, 27]:
	env = gym.make('TicTacToe-logic-dim2-v1', symbols=[-1, 1], board_size=3, win_size=3)
elif config in [28, 29]:
	env = gym.make('TicTacToe-logic-dim2-v3', symbols=[-1, 1], board_size=3, win_size=3)
else:
	env = gym.make('TicTacToe-plain-v0', symbols=[-1, 1], board_size=3, win_size=3)

env_seed = 111 # reproducible, general Policy gradient has high variance
env.seed(env_seed)

if config in [0, 1]:
	RL = Qtable(
		action_dim = env.action_space.n,
		state_dim = env.state_space.shape[0],
		learning_rate = 0.8,
		gamma = 0.9,	# doesn't matter for gym TicTacToe
	)
elif config in [20, 21, 22, 23, 24]:
	RL = DQN(
		action_dim = env.action_space.n,
		state_dim = env.state_space.shape[0],
		learning_rate = 0.001,
		gamma = 0.9,	# doesn't matter for gym TicTacToe
	)
elif config == 25:
	RL = DQN(
		action_dim = env.action_space.n,
		state_dim = env.state_space.shape[0],	# ignored, using dim=2
		learning_rate = 0.001,
		gamma = 1.0,	# seems to matter for looping!
	)
elif config in [26, 27, 28, 29]:
	RL = DQN(
		action_dim = env.action_space.n,
		state_dim = env.state_space.shape[0],
		learning_rate = 0.001,
		gamma = 0.9,	# doesn't matter for gym TicTacToe
	)
elif config == 15:
	RL = SAC(
		action_dim = 1, # env.action_space.n,
		state_dim = env.state_space.shape[0],
		learning_rate = 0.001,
		gamma = 0.9,	# doesn't matter for gym TicTacToe
	)
else:
	RL = PolicyGradient(
		n_actions = env.action_space.n,
		n_features = env.state_space.shape[0],
		learning_rate = 0.001,
		gamma = 0.9,	# doesn't matter for gym TicTacToe
	)

from datetime import datetime
startTime = datetime.now()
timeStamp = startTime.strftime("%d-%m-%Y(%H:%M)")

topology, num_weights = RL.net_info()
tag += "." + topology
log_name = "results/results." + tag + "." + timeStamp + ".txt"
log_file = open(log_name, "w+")
print("Log file opened:", log_name)

print("action_space =", env.action_space)
print("n_actions =", env.action_space.n)
print("state_space =", env.state_space)
print("n_features = state dim =", env.state_space.shape[0])
print("state_space.high =", env.state_space.high)
print("state_space.low =", env.state_space.low)

# import sys
for f in [log_file, sys.stdout]:
	f.write("# Config # = " + str(config) + '\n')
	f.write("# Model = " + tag + '\n')
	f.write("# Gym = " + env.spec.id + '\n')
	f.write("# algorithm = " + RL.__module__ + '\n')
	f.write("# Num weights = " + str(num_weights) + '\n')
	f.write("# Learning rate = " + str(RL.lr) + '\n')
	f.write("# Env random seed = " + str(env_seed) + '\n')
	f.write("# Start time: " + timeStamp + '\n')

# **** This is for catching warnings and to debug them:
# import warnings
# warnings.filterwarnings("error")

import signal
import readline
print("Press Ctrl-C to pause and execute your own Python code\n")

model_name = "model." + tag

command = None

def ctrl_C_handler(sig, frame):
	# global model_name
	global command
	print("\n\x1b[0m **** program paused ****")
	print("Enter Python code, or:\n'q' to exit\n'' to continue\n's' to save model\n'g' to play game\n'v' to visualize Q values")
	command = input(">>> ")
	if command == "q":
		command = "log_file.close(); exit(0)"
	elif command == '':
		command = None
	elif command == 'g':
		command = "play_1_game_with_human()"
	elif command == 'v':
		command = "visualize_Q()"
	elif command == 's':
		command = "RL.save_net(model_name + '.' + timeStamp)"
	# Other commands will be executed in the main loop, see below
	"""
	print("Enter filename to save network to file")
	print("Default file: ", model_name + "." + timeStamp)
	print("Enter 'x' to exit")
	model_name = input() or model_name
	if model_name == "x":
		log_file.close()
		exit(0)
	else:
		if config == 1 or config == 2:
			RL.save_net(model_name + "." + timeStamp)
		else:
			print("Save model not implemented yet.")
	"""

signal.signal(signal.SIGINT, ctrl_C_handler)

# Check if text-to-speech software Ekho is installed:
from subprocess import call, DEVNULL
speech = True
try:
	# print("Testing [optional] Ekho version =")
	call(['ekho', '--version'], stdout=DEVNULL)
except FileNotFoundError:
	speech = False

import glob
if config == 3 or config == 4:		# TensorFlow
	files = glob.glob("TensorFlow_models/" + model_name + "*.index")
elif config == 0:
	files = glob.glob("*.npy")
else:
	files = glob.glob("PyTorch_models/" + model_name + "*.dict")
files.sort()
for i, fname in enumerate(files):
	if i % 2:
		print(end="\x1b[32m")
	else:
		print(end="\x1b[0m")
	if config == 4 or config == 5:		# TensorFlow
		print("%2d %s" %(i, fname[24:-6]))
	elif config == 0:
		print("%2d %s" %(i, fname))
	else:
		print("%2d %s" %(i, fname[21:-5]))
print(end="\x1b[0m")
j = input("Load model? (Enter number or none): ")
if j:
	if config == 4 or config == 5:		# TensorFlow
		RL.load_net(files[int(j)][18:-11])
	elif config == 0:
		RL.load_net(files[int(j)])
	else:
		RL.load_net(files[int(j)][15:-5])

def preplay_moves():
	return
	global state
	state, _, _, _ = env.step(0, -1)
	state, _, _, _ = env.step(3, 1)
	state, _, _, _ = env.step(6, -1)
	state, _, _, _ = env.step(4, 1)
	# state, _, _, _ = env.step(5, -1)
	# state, _, _, _ = env.step(1, 1)
	return

print("Pre-play moves:")
state = env.reset()
preplay_moves()
env.render(mode=None)

# hyper-parameters
batch_size   = 256
# max_episodes = 40
# max_steps    = 150	# Pendulum needs 150 steps per episode to learn well, cannot handle 20
# frame_idx    = 0
# explore_steps = 0
rewards      = []
reward_scale = 10.0
model_path   = './model/sac'

import websockets
from websockets.sync.client import connect
import json
import TTT_utils

def visualize_Q():
	global INTERMEDIATE
	with connect("ws://localhost:5678") as websocket:
		print("Visualizing Q values...")
		while True:
		# the state 's' is as entered by web GUI
			# get board vector from GUI Javascript
			board, memory = json.loads(websocket.recv())
			if board == 0:		# exit signal
				break
			if INTERMEDIATE:
				print("state=", board, '\t', memory)
				logits, probs = RL.visualize_q(board, memory)
			else:
				print("state=", board)
				logits, probs = RL.visualize_q(board)
			if probs is not None:
				print("probs=", end=' ')
				for p in probs:
					print('{:.4f}'.format(p), end=' ')
				websocket.send(json.dumps(["probs", probs]))
				websocket.send(json.dumps(["Q-vals", logits]))

def play_1_game_with_human():
	state = env.reset()
	preplay_moves()
	done = False
	user = -1
	while not done:
		env.render()
		if user == -1:
			print("X's move =", end='')		# will be printed by RL.choose_action()
			action1 = RL.choose_action(state)
			state1, reward1, done, rtype = env.step(action1, -1)
			if done:
				state = state1
				reward1 = reward2 = 0
		elif user == 1:			# human player
			action2 = int(input("Your move (0-8)? "))
			state2, reward2, done, rtype = env.step(action2, 1)
			r_x = reward1		# reward w.r.t. player X = AI
			if reward2 > 19.0:
				r_x -= 20.0
			elif reward2 > 9.0:	# draw: both players +10
				r_x += 10.0
			state = state2
			reward1 = reward2 = 0

		# If the game isn't over, change the current player
		if not done:
			if rtype != 'thinking':
				user = -user
	env.render()
	print("*** GAME OVER ***")

train_once = False		# you may use Ctrl-C to change this
DETERMINISTIC = False
INTERMEDIATE = False	# whether the state has intermediate thoughts
RENDER = 0
# **** RENDER flag:
# bit 1 (1) = display every real action
# bit 2 (2) = display only end board state
# bit 3 (4) = display every action including intermediate thoughts
RENDERMODE = "HTML"
i_episode = 0
running_reward = 0.0

while True:
	i_episode += 1
	state = env.reset()
	print("**** state =", state)
	preplay_moves()
	if RENDER > 0:
		env.render(mode=RENDERMODE)
	done = False
	user = -1
	reward1 = 0
	reward2 = 0

	while not done:

		if user == -1:		# AI player
			# action is integer 0...8
			action1 = RL.choose_action(state)
			if RENDER & 4:
				with connect("ws://localhost:5678") as websocket:
					websocket.send(json.dumps(action1.item()))
			state1, reward1, done, rtype = env.step(action1, -1)
			if done:		# otherwise wait for random player to react
				RL.replay_buffer.push(state, action1, reward1, RL.endState, done)	# state1 = endState
		elif user == 1:		# random player
			# NOTE: random player never chooses occupied squares
			action2 = RL.play_random(state1, env.action_space)
			state2, reward2, done, rtype = env.step(action2, 1)
			r_x = reward1		# reward w.r.t. player X = AI
			# **** Scoring: AI win > draw > lose > crash
			#                +20      +10   -20    -30
			if reward2 > 19.0:
				r_x -= 20.0
			elif reward2 > 9.0:	# draw: both players +10
				r_x += 10.0
			if done:
				RL.replay_buffer.push(state, action1, r_x, RL.endState, done)
			else:
				RL.replay_buffer.push(state, action1, r_x, state2, done)
			state = state2

		# If the game isn't over, change the current player
		if not done:
			if rtype != 'thinking':
				user = -1 if user == 1 else 1
			if RENDER & 1:
				env.render(mode = RENDERMODE)
		elif RENDER & 3:
			# await asyncio.sleep(0.1)
			env.render(mode = RENDERMODE)

	# **** Game ended:
	per_game_reward = RL.replay_buffer.last_reward()		# actually only the last reward is non-zero, for gym TicTacToe
	if per_game_reward > -0.5:
		color = '\x1b[32m'			# green
	if per_game_reward < -19.0:
		color = '\x1b[33m'			# yellow
	if per_game_reward < -26.0:
		color = '\x1b[31m'			# red
	print(color + str(per_game_reward), end=' ')
	# print("rtype=", rtype)
	# if input("! to break ==>") == '!':
		# break

	# The replay buffer seems to record every game and it keeps growing
	# how to get the reward of the last game only?
	# the running reward formula below is correct, if per_game_reward is correct

	running_reward = running_reward * 0.97 + per_game_reward * 0.03

	if len(RL.replay_buffer) > batch_size:
		_ = RL.update(batch_size, reward_scale)

	if command:				# wait till end-of-game now to execute command
		try:
			exec(command)
		except Exception as e:
			print("Exception:")
			print(e)
		finally:
			command = None

	if i_episode % 100 == 0:
		if RENDER > 0:
			with connect("ws://localhost:5678") as websocket:
				websocket.send(json.dumps('ask'))
				RENDER = json.loads(websocket.recv())
				# print("RENDER=", RENDER)

		rr = round(running_reward, 5)
		print("\n\t\x1b[0m", i_episode, "Running reward:", "\x1b[32m" if rr >= 0.0 else "\x1b[31m", rr, "\x1b[0m")	#, "lr =", RL.lr)
		if INTERMEDIATE:
			print("good:rational =", env.good, ":", env.rational, ":", env.irrational)
			env.good = 0
			env.rational = 0
			env.irrational = 0
		# RL.set_learning_rate(i_episode)
		log_file.write(str(i_episode) + ' ' + str(running_reward) + '\n')
		log_file.flush()

		if i_episode % 1000 == 0:
			if rr < 0.0:
				s = 'minus ' + str(int(-rr))
			else:
				s = str(int(rr))
			if speech:
				call(['ekho', s, '-v', 'English', '--english-speed', '20'])

			delta = datetime.now() - startTime
			print('[ {d}d {h}:{m}:{s} ]'.format(d=delta.days, h=delta.seconds//3600, m=(delta.seconds//60)%60, s=delta.seconds%60))

			if i_episode == 200000:		# approx 1 hours' run for pyTorch, half hour for TensorFlow
				# print('\007')	# sound beep
				endTime = datetime.now()
				endStamp = endTime.strftime("%d-%m-%Y(%H:%M)")
				log_file.write("# End time: " + endStamp + '\n')
				if speech:
					call(['ekho', '新档案', '-s=-20'])	# speak "new file"
				log_file.close()
				RL.save_net(model_name + "." + timeStamp)
				if train_once:
					break

				# Preferable to get a new time stamp now:
				startTime = datetime.now()
				timeStamp = startTime.strftime("%d-%m-%Y(%H:%M)")
				i_episode = 0
				log_name = "results/results." + tag + "." + timeStamp + ".txt"
				log_file = open(log_name, "w+")

				print("New log file opened:", log_name)
		else:
			call(['play', '-n', '-q', 'synth', '0.05', 'sine', '2300', 'gain', '-25'])

endTime = datetime.now()
timeStamp = endTime.strftime("%d-%m-%Y(%H:%M)")
log_file.write("# End time: " + timeStamp + '\n')
print('\007')	# sound beep
