"""
Tic Tac Toe with Policy Gradient

With a choice of engines:
- PyTorch: 1.9.0+cpu
- TensorFlow: 2.0

With a choice of representations:
- fully-connected
- symmetric NN 
"""

print("1. PyTorch\t symmetric NN")
print("2. PyTorch\t fully-connected NN")
print("3. TensorFlow\t symmetric NN")
print("4. TensorFlow\t fully-connected NN")
config = int(input("Choose config: "))

import gym

if config == 1:
	from RL_symNN_pyTorch import PolicyGradient
	tag = "symNN.pyTorch"
elif config == 2:
	from RL_full_pyTorch import PolicyGradient
	tag = "full.pyTorch"
elif config == 3:
	from RL_symNN_TensorFlow import PolicyGradient
	tag = "symNN.TensorFlow"
elif config == 4:
	from RL_full_TensorFlow import PolicyGradient
	tag = "full.TensorFlow"

DISPLAY_REWARD_THRESHOLD = 19.90  # renders environment if total episode reward > threshold
RENDER = False  # rendering wastes time

import gym_tictactoe
if config == 1 or config == 3:
	env = gym.make('TicTacToe-logic-v0', symbols=[-1, 1], board_size=3, win_size=3)
else:
	env = gym.make('TicTacToe-plain-v0', symbols=[-1, 1], board_size=3, win_size=3)

env_seed = 113 # reproducible, general Policy gradient has high variance
env.seed(env_seed)

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
log_name = "results." + tag + "." + timeStamp + ".txt"
log_file = open(log_name, "w+")
print("Log file opened:", log_name)

# print("action_space =", env.action_space)
# print("n_actions =", env.action_space.n)
# print("state_space =", env.state_space)
# print("n_features =", env.state_space.shape[0])
# print("state_space.high =", env.state_space.high)
# print("state_space.low =", env.state_space.low)

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
	print("\n **** program paused ****")
	print("Enter your code (! to exit)")
	command = input(">>> ")
	if command == '!':
		log_file.close()
		exit(0)
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
files = glob.glob("TensorFlow_models/" + model_name + "*.index")
files.sort()
for i, fname in enumerate(files):
	if i % 2:
		print(end="\x1b[32m")
	else:
		print(end="\x1b[0m")
	print("%2d %s" %(i, fname[24:-6]))
print(end="\x1b[0m")
j = input("Load model? (Enter number or none): ")
if j:
	RL.load_net(files[int(j)][9:-11])

train_once = False		# you may use Ctrl-C to change this
i_episode = 0
while True:
	i_episode += 1
	state = env.reset()

	done = False
	user = 0
	reward1 = reward2 = 0

	# ***** Pre-play moves:
	state, _, _, _ = env.step(0, -1)
	state, _, _, _ = env.step(3, 1)
	state, _, _, _ = env.step(6, -1)
	state, _, _, _ = env.step(4, 1)
	state, _, _, _ = env.step(5, -1)
	state, _, _, _ = env.step(1, 1)
	while not done:

		if user == 0:
			action1 = RL.choose_action(state)
			state1, reward1, done, infos = env.step(action1, -1)
			if done:
				RL.store_transition(state, action1, reward1)
				state = state1
				reward1 = reward2 = 0
		elif user == 1:
			# NOTE: random player never chooses occupied squares
			action2 = RL.play_random(state1, env.action_space)
			state2, reward2, done, infos = env.step(action2, 1)
			RL.store_transition(state, action1, reward1 - reward2)		# not r1 + r2 as rewards cancel out each other
			state = state2
			reward1 = reward2 = 0

		# If the game isn't over, change the current player
		if not done:
			user = 0 if user == 1 else 1

	# **** Game ended:
	# print(RL.ep_rs)
	per_game_reward = sum(RL.ep_rs)		# actually only the last reward is non-zero, for gym TicTacToe

	if 'running_reward' not in globals():
		running_reward = per_game_reward
	else:
		running_reward = running_reward * 0.99 + per_game_reward * 0.01
	if running_reward > DISPLAY_REWARD_THRESHOLD:
		RENDER = True     # rendering

	if i_episode % 100 == 0:
		rr = round(running_reward,5)
		print(i_episode, "running reward:", "\x1b[32m" if rr >= 0.0 else "\x1b[31m", rr, "\x1b[0m")	#, "lr =", RL.lr)
		# RL.set_learning_rate(i_episode)
		log_file.write(str(i_episode) + ' ' + str(running_reward) + '\n')
		log_file.flush()

		if command:
			try:
				exec(command)
			except Exception as e:
				print("Exception:")
				print(e)
			finally:
				command = None

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
				log_name = "results." + tag + "." + timeStamp + ".txt"
				log_file = open(log_name, "w+")
				print("New log file opened:", log_name)

	RL.learn()

print('\007')	# sound beep
