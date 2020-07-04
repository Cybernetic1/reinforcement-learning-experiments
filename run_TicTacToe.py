"""
Policy Gradient, Reinforcement Learning.

Tic-Tac-Toe test

Using:
Tensorflow: 2.0
gym: 0.8.0
"""

import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 300  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

import gym_tictactoe
env = gym.make('TicTacToe-v2', symbols=[-1, 1], board_size=3, win_size=3)

user = 0
done = False
reward = 0

# Reset the env before playing
state = env.reset()

env.seed(1)     # reproducible, general Policy gradient has high variance

print(env.action_space)
print(env.state_space)
print(env.state_space.high)
print(env.state_space.low)

RL = PolicyGradient(
	n_actions=env.action_space.n,
	n_features=env.state_space.shape[0],
	learning_rate=0.002,
	reward_decay=0.98,
	# output_graph=True,
)

print("n_features=", RL.n_features)

for i_episode in range(30000):
	state = env.reset()

	done = False
	while not done:			# loop per game
		action = RL.choose_action(state)			# this is an int ∈ {0,...,8}

		if user == 0:
			state_, reward, done, infos = env.step(action, -1)
		elif user == 1:
			while True:
				random_act = env.action_space.sample()
				if (random_act + 1) in state or -(random_act + 1) in state:
					continue
				break
			state_, reward, done, infos = env.step(random_act, 1)

		RL.store_transition(state, action, reward)

		# env.render(mode=None)
		# print("state vec =", state_)

		# If the game isn't over, change the current player
		if not done:
			user = 0 if user == 1 else 1
		else:
			ep_rs_sum = sum(RL.ep_rs)

			if 'running_reward' not in globals():
				running_reward = ep_rs_sum
			else:
				running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
			if running_reward > DISPLAY_REWARD_THRESHOLD:
				RENDER = True     # rendering

			if i_episode % 100 == 0:
				print("episode:", i_episode, " running reward:", int(running_reward))

			vt = RL.learn()

			if i_episode == -1:
				plt.plot(vt)    # plot the episode vt
				plt.xlabel('episode steps')
				plt.ylabel('normalized state-action value')
				plt.show()

			# if reward == 10:
				# print("Draw !")
			# elif reward == -20:
				# print("Infos : " + str(infos))
				# if user == 0:
					# print("Random wins ! AI Reward : " + str(reward))
				# elif user == 1:
					# print("AI wins ! AI Reward : " + str(-reward))
			# elif reward == 20:
				# if user == 0:
					# print("AI wins ! AI Reward : " + str(reward))
				# elif user == 1:
					# print("Random wins ! AI Reward : " + str(reward))

		state = state_
