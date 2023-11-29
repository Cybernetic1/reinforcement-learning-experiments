"""
Tic Tac Toe with Policy Gradient

With a choice of engines:
- PyTorch: 1.9.0+cpu
- TensorFlow: 2.0

With a choice of representations:
- fully-connected
- symmetric NN
"""

print("0. Python\tQ-table")
print("1. PyTorch\tDQN")
print("2. PyTorch\tPG\tsymmetric NN")
print("3. PyTorch\tPG\tfully-connected NN")
print("4. TensorFlow\tPG\tsymmetric NN")
print("5. TensorFlow\tPG\tfully-connected NN")
print("6. PyTorch\tPG\tTransformer")
print("7. PyTorch\tSAC\tfully-connected NN")
print("8. PyTorch\tSAC\tTransformer")
config = int(input("Choose config: ") or '1')

import gym

if config == 0:
	from RL_Qtable import Qtable
	tag = "Qtable"
if config == 1:
	from RL_DQN_pyTorch import DQN
	tag = "DQN"
if config == 2:
	from RL_symNN_pyTorch import PolicyGradient
	tag = "symNN.pyTorch"
elif config == 3:
	from RL_full_pyTorch import PolicyGradient
	tag = "full.pyTorch"
elif config == 4:
	from RL_symNN_TensorFlow import PolicyGradient
	tag = "symNN.TensorFlow"
elif config == 5:
	from RL_full_TensorFlow import PolicyGradient
	tag = "full.TensorFlow"
elif config == 6:
	from RL_Transformer_pyTorch import PolicyGradient
	tag = "Transformer.pyTorch"
elif config == 7:
	from SAC_full_pyTorch import SAC, ReplayBuffer
	tag = "SAC.full.pyTorch"
elif config == 8:
	from SAC_Transformer_pyTorch import SAC, ReplayBuffer
	tag = "SAC.Transformer.pyTorch"

import gym_tictactoe
if config in [2, 4, 6]:
	env = gym.make('TicTacToe-logic-v0', symbols=[-1, 1], board_size=3, win_size=3)
else:
	env = gym.make('TicTacToe-plain-v0', symbols=[-1, 1], board_size=3, win_size=3)

env_seed = 111 # reproducible, general Policy gradient has high variance
env.seed(env_seed)

if config == 0:
	RL = Qtable(
		action_dim = env.action_space.n,
		state_dim = env.state_space.shape[0],
		learning_rate = 0.001,
		gamma = 0.9,	# doesn't matter for gym TicTacToe
	)
if config == 1:
	RL = DQN(
		action_dim = env.action_space.n,
		state_dim = env.state_space.shape[0],
		learning_rate = 0.001,
		gamma = 0.9,	# doesn't matter for gym TicTacToe
	)
elif config >= 7:
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
print("n_features =", env.state_space.shape[0])
print("state_space.high =", env.state_space.high)
print("state_space.low =", env.state_space.low)

import sys
for f in [log_file, sys.stdout]:
	f.write("# Model = " + tag + '\n')
	f.write("# Num weights = " + str(num_weights) + '\n')
	f.write("# Learning rate = " + str(RL.lr) + '\n')
	f.write("# Env random seed = " + str(env_seed) + '\n')
	f.write("# Start time: " + timeStamp + '\n')

# **** This is for catching warnings and to debug them:
# import warnings
# warnings.filterwarnings("error")

import signal
print("Press Ctrl-C to pause and execute your own Python code\n")

model_name = "model." + tag

command = None

def ctrl_C_handler(sig, frame):
	# global model_name
	global command
	print("\n\x1b[0m **** program paused ****")
	print("Enter Python code (return to exit, c to continue, s to save model, g to play game)")
	command = input(">>> ")
	if command == "":
		log_file.close()
		exit(0)
	elif command == 'c':
		pass
	elif command == 'g':
		command = "play_1_game_with_human()"
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

import glob
if config == 3 or config == 4:		# TensorFlow
	files = glob.glob("TensorFlow_models/" + model_name + "*.index")
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
	else:
		print("%2d %s" %(i, fname[21:-5]))
print(end="\x1b[0m")
j = input("Load model? (Enter number or none): ")
if j:
	if config == 4 or config == 5:		# TensorFlow
		RL.load_net(files[int(j)][18:-11])
	else:
		RL.load_net(files[int(j)][15:-5])

def preplay_moves():
	return
	state, _, _, _ = env.step(0, -1)
	state, _, _, _ = env.step(3, 1)
	state, _, _, _ = env.step(6, -1)
	state, _, _, _ = env.step(4, 1)
	# state, _, _, _ = env.step(5, -1)
	# state, _, _, _ = env.step(1, 1)
	return

import asyncio
print("Pre-play moves:")
state = env.reset()
preplay_moves()
# asyncio.get_event_loop().run_until_complete(env.render())
# await env.render()
# asyncio.run(env.render())
env.render()
# env.render()

# hyper-parameters
batch_size   = 512
# max_episodes = 40
# max_steps    = 150	# Pendulum needs 150 steps per episode to learn well, cannot handle 20
# frame_idx    = 0
# explore_steps = 0
rewards      = []
reward_scale = 10.0
model_path   = './model/sac'

from subprocess import call

def play_1_game_with_human():
	state = env.reset()
	preplay_moves()
	done = False
	user = 0
	while not done:
		env.render()
		if user == 0:
			print("X's move =", end='')			# will be printed by RL.choose_action()
			action1 = RL.choose_action(state)
			state1, reward1, done, _ = env.step(action1, -1)
			if done:
				state = state1
				reward1 = reward2 = 0
		elif user == 1:			# human player
			action2 = int(input("Your move (0-8)? "))
			state2, reward2, done, _ = env.step(action2, 1)
			r_x = reward1		# reward w.r.t. player X = AI
			if reward2 > 19.0:
				r_x -= 20.0
			elif reward2 > 9.0:	# draw: both players +10
				r_x += 10.0
			state = state2
			reward1 = reward2 = 0

		# If the game isn't over, change the current player
		if not done:
			user = 0 if user == 1 else 1
	env.render()
	RL.clear_data()
	print("*** GAME OVER ***")

train_once = False		# you may use Ctrl-C to change this
DETERMINISTIC = False
RENDER = 1
i_episode = 0
running_reward = 0.0

import websockets

while True:
	i_episode += 1
	state = env.reset()
	preplay_moves()

	done = False
	user = -1
	reward1 = reward2 = 0

	while not done:

		if user == -1:		# AI player
			# action is integer 0...8
			action1 = RL.choose_action(state)
			state1, reward1, done, _ = env.step(action1, -1)
			if done:
				RL.replay_buffer.push(state, action1, reward1, state1, done)
		elif user == 1:		# random player
			# NOTE: random player never chooses occupied squares
			action2 = RL.play_random(state1, env.action_space)
			state2, reward2, done, _ = env.step(action2, 1)
			r_x = reward1		# reward w.r.t. player X = AI
			# **** Scoring: AI win > draw > lose > crash
			#                +20      +10   -20    -30
			if reward2 > 19.0:
				r_x -= 20.0
			elif reward2 > 9.0:	# draw: both players +10
				r_x += 10.0
			RL.replay_buffer.push(state, action1, r_x, state2, done)
			state = state2

		# If the game isn't over, change the current player
		if not done:
			user = -1 if user == 1 else 1
		elif RENDER > 0:
			# await asyncio.sleep(0.1)
			env.render(mode = 'HTML')
		if RENDER == 2:
			env.render(mode = 'HTML')

	# **** Game ended:
	per_game_reward = RL.replay_buffer.last_reward()		# actually only the last reward is non-zero, for gym TicTacToe
	if per_game_reward > -0.5:
		color = '\x1b[32m'			# green
	if per_game_reward < -19.0:
		color = '\x1b[33m'			# yellow
	if per_game_reward < -21.0:
		color = '\x1b[31m'			# red
	print(color + str(per_game_reward), end=' ')
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
		call(['play', '-n', '-q', 'synth', '0.05', 'sine', '2300', 'gain', '-20'])
		rr = round(running_reward, 5)
		print("\n\t\x1b[0m", i_episode, "Running reward:", "\x1b[32m" if rr >= 0.0 else "\x1b[31m", rr, "\x1b[0m")	#, "lr =", RL.lr)
		# RL.set_learning_rate(i_episode)
		log_file.write(str(i_episode) + ' ' + str(running_reward) + '\n')
		log_file.flush()

		if i_episode % 1000 == 0:
			delta = datetime.now() - startTime
			print('[ {d}d {h}:{m}:{s} ]'.format(d=delta.days, h=delta.seconds//3600, m=(delta.seconds//60)%60, s=delta.seconds%60))

			if i_episode == 200000:		# approx 1 hours' run for pyTorch, half hour for TensorFlow
				print('\007')	# sound beep
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

print('\007')	# sound beep
