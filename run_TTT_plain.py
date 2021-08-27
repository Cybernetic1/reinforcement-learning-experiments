"""
Policy Gradient, Reinforcement Learning.

Tic-Tac-Toe test

Using:
PyTorch: 1.9.0+cpu
gym: 0.8.0
"""

import datetime

import gym
from RL_plain import PolicyGradient

import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd

DISPLAY_REWARD_THRESHOLD = 300  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

import gym_tictactoe
env = gym.make('TicTacToe-plain-v0', symbols=[-1, 1], board_size=3, win_size=3)
env.seed(1)     # reproducible, general Policy gradient has high variance

print("env.action_space:", env.action_space)
print("env.state_space:", env.state_space)

RL = PolicyGradient(
	n_actions=env.action_space.n,
	n_features=env.state_space.shape[0],
	learning_rate = 0.01,
	gamma = 0.98,
	# output_graph=True,
)

now = datetime.datetime.now()
print ("Start Time =", now.strftime("%Y-%m-%d %H:%M:%S"))

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
			while True:
				random_act = env.action_space.sample()
				if state1[random_act] == 0:
					break
			state2, reward2, done, infos = env.step(random_act, 1)
			RL.store_transition(state, action1, reward1 - reward2)		# why is it r1 + r2? wouldn't the rewards cancel out each other? 
			state = state2
			reward1 = reward2 = 0

		# env.render(mode=None)

		# If the game isn't over, change the current player
		if not done:
			user = 0 if user == 1 else 1

	ep_rs_sum = sum(RL.ep_rs)

	if 'running_reward' not in globals():
		running_reward = ep_rs_sum
	else:
		running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
	if running_reward > DISPLAY_REWARD_THRESHOLD:
		RENDER = True     # rendering

	if i_episode % 100 == 0:
		rr = int(running_reward)
		print(i_episode, "running reward:", "\x1b[32m" if rr >= 0 else "\x1b[31m", rr, "\x1b[0m")	#, "lr =", RL.lr)
		# RL.set_learning_rate(i_episode)

	if i_episode % 1000 == 0:
		now = datetime.datetime.now()
		print (now.strftime("%Y-%m-%d %H:%M:%S"))

	vt = RL.learn()
