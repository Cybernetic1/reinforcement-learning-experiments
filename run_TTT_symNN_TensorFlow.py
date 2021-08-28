"""
Policy Gradient, Reinforcement Learning.

Tic-Tac-Toe test

Using:
Tensorflow: 2.0
gym: 0.8.0
"""

import gym
from RL_symNN_TensorFlow import PolicyGradient

DISPLAY_REWARD_THRESHOLD = 19.90  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

import gym_tictactoe
env = gym.make('TicTacToe-logic-v0', symbols=[-1, 1], board_size=3, win_size=3)
env.seed(777)     # reproducible, general Policy gradient has high variance

RL = PolicyGradient(
	n_actions=env.action_space.n,
	n_features=env.state_space.shape[0],
	learning_rate=0.002,
	reward_decay=0.9,		# seems useless for gym TicTacToe
	# output_graph=True,
)

print("\nParameters:")
print("action_space =", env.action_space)
print("n_actions =", env.action_space.n)
print("state_space =", env.state_space)
print("n_features =", env.state_space.shape[0])
print("state_space.high =", env.state_space.high)
print("state_space.low =", env.state_space.low)
print("learning rate =", RL.lr)

from datetime import datetime
startTime = datetime.now()
timeStamp = startTime.strftime("%d-%m-%Y(%H:%M)")
print ("\nStart Time =", timeStamp)

fname = "results.symNN.TensorFlow." + timeStamp + ".txt"
log_file = open(fname, "a+")
print("Log file opened:", fname)

i_episode = 0
while True:
	i_episode += 1
	state = env.reset()

	done = False
	user = 0
	reward1 = reward2 = 0
	while not done:

		if user == 0:		# AI player
			action1 = RL.choose_action(state)
			state1, reward1, done, infos = env.step(action1, -1)
			if done:
				RL.store_transition(state, action1, reward1)
				state = state1
				reward1 = reward2 = 0
		elif user == 1:		# random player
			while True:		# random player will never choose occupied squares
				random_act = env.action_space.sample()
				x = random_act % 3
				y = random_act // 3
				occupied = False
				for i in range(0, 27, 3):		# scan through all 9 propositions, each proposition is a 3-vector
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
			RL.store_transition(state, action1, reward1 - reward2)	# refer to ttt-test.py for sign of 'reward'
			state = state2
			reward1 = reward2 = 0

		# env.render(mode=None)
		# print("state vec =", state)

		# If the game isn't over, change the current player
		if not done:
			user = 0 if user == 1 else 1

	# **** Game over:
	ep_rs_sum = sum(RL.ep_rs)

	if 'running_reward' not in globals():
		running_reward = ep_rs_sum
	else:
		running_reward = running_reward * 0.8 + ep_rs_sum * 0.2
	if running_reward >= DISPLAY_REWARD_THRESHOLD:
		RENDER = True     # rendering

	if i_episode % 100 == 0:
		rr = round(running_reward,5)
		print(i_episode, "running reward:", "\x1b[32m" if rr >= 0.0 else "\x1b[31m", rr, "\x1b[0m")	#, "lr =", RL.lr)
		# RL.set_learning_rate(i_episode)
		log_file.write(str(i_episode) + ' ' + str(rr) + '\n')
		log_file.flush()

		if i_episode % 1000 == 0:
			delta = datetime.now() - startTime
			# print (now.strftime("%Y-%m-%d %H:%M:%S"))
			print('[ {d}d {h}:{m}:{s} ]'.format(d=delta.days, h=delta.seconds//3600, m=(delta.seconds//60)%60, s=delta.seconds%60))

			if i_episode == 200000:		# approx 1 hours' run
				log_file.close()
				# RL.save_net(fname)	# not implemented yet
				sys.exit(0)

	vt = RL.learn()
