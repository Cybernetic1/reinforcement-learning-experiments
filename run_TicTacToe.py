"""
Tic Tac Toe with Policy Gradient

with a choice of configs:
PyTorch: 1.9.0+cpu
TensorFlow: 2.0
"""

print("1. PyTorch\t Symmetric NN")
print("2. PyTorch\t fully-connected NN")
print("3. TensorFlow\t Symmetric NN")
print("4. TensorFlow\t fully-connected NN")
config = int(input("Choose config: "))

import gym

if config == 1:
	from RL_symNN_pyTorch import PolicyGradient
	tag = "symNN.pyTorch."
elif config == 2:
	from RL_full_pyTorch import PolicyGradient
	tag = "full.pyTorch."
elif config == 3:
	from RL_symNN_TensorFlow import PolicyGradient
	tag = "symNN.TensorFlow."
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
env.seed(777)     # reproducible, general Policy gradient has high variance

from datetime import datetime
startTime = datetime.now()
timeStamp = startTime.strftime("%d-%m-%Y(%H:%M)")

fname = "results." + tag + timeStamp + ".txt"
log_file = open(fname, "a+")
print("Log file opened:", fname)

RL = PolicyGradient(
	n_actions=env.action_space.n,
	n_features=env.state_space.shape[0],
	learning_rate = 0.001,
	gamma = 0.9,	# doesn't matter for gym TicTacToe
	# output_graph=True,
)

# print("action_space =", env.action_space)
# print("n_actions =", env.action_space.n)
# print("state_space =", env.state_space)
# print("n_features =", env.state_space.shape[0])
# print("state_space.high =", env.state_space.high)
# print("state_space.low =", env.state_space.low)

import sys
for f in [log_file, sys.stdout]:
	f.write("# Model = " + tag + '\n')
	f.write("# Learning rate =" + str(RL.lr) + '\n')
	f.write("# Start time =" + timeStamp + '\n')

# **** This is for catching warnings and to debug them:
# import warnings
# warnings.filterwarnings("error")

import signal
print("Press Ctrl-C to pause and optionally save network to file\n")

def ctrl_C_handler(sig, frame):
	print("\n **** program paused ****")
	fname = "model." + tag + ".dict"
	print("Enter filename (default: {s}) to save network to file".format(s=fname))
	print("Enter 'x' to exit")
	fname = input() or fname
	if fname == "x":
		log_file.close()
		exit(0)
	else:
		if config == 1 or config == 2:
			RL.save_net(fname)
		else:
			print("Save model not implemented yet.")

signal.signal(signal.SIGINT, ctrl_C_handler)

i_episode = 0
while True:
	i_episode += 1
	state = env.reset()

	done = False
	user = 0
	reward1 = reward2 = 0
	while not done:

		if user == 0:
			action1 = RL.choose_action(state)
			state1, reward1, done, infos = env.step(action1, -1)
			if done:
				RL.store_transition(state, action1, reward1)
				state = state1
				reward1 = reward2 = 0
		elif user == 1:
			while True:		# random player will never choose occupied squares
				random_act = env.action_space.sample()
				x = random_act % 3
				y = random_act // 3
				occupied = False
				for i in range(0, 27, 3):		# scan through all 9 propositions, each proposition is a 3-vector
					# 'proposition' is a numpy array[3]
					proposition = state1[i : i + 3]
					# print("proposition=",proposition)
					if ([x,y,1] == proposition).all():
						occupied = True
						break
					if ([x,y,-1] == proposition).all():
						occupied = True
						break
				if not occupied:
					break

			state2, reward2, done, infos = env.step(random_act, 1)
			RL.store_transition(state, action1, reward1 - reward2)		# why is it r1 + r2? wouldn't the rewards cancel out each other? 
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

		if i_episode % 1000 == 0:
			delta = datetime.now() - startTime
			print('[ {d}d {h}:{m}:{s} ]'.format(d=delta.days, h=delta.seconds//3600, m=(delta.seconds//60)%60, s=delta.seconds%60))

			if i_episode == 200000:		# approx 1 hours' run for pyTorch, half hour for TensorFlow
				log_file.close()
				if config == 1 or config == 2:
					RL.save_net(fname)
				exit(0)

	vt = RL.learn()
