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

DISPLAY_REWARD_THRESHOLD = 20  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

import gym_tictactoe
env = gym.make('TicTacToe-v2', symbols=[-1, 1], board_size=3, win_size=3)
env.seed(1)     # reproducible, general Policy gradient has high variance

print(env.action_space)
print(env.state_space)
print(env.state_space.high)
print(env.state_space.low)

RL = PolicyGradient(
	n_actions=env.action_space.n,
	n_features=env.state_space.shape[0],
	# learning_rate=0.002,
	reward_decay=0.9,
	# output_graph=True,
)

print("n_features=", RL.n_features)

i_episode = 0
# for i_episode in range(60000):
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
				x = random_act % 3
				y = random_act // 3
				found = False
				for i in range(0, 27, 3):
					chunk = state1[i : i + 3]
					# print("chunk=",chunk)
					if ([x,y,1] == chunk).all():
						found = True
						break
					if ([x,y,-1] == chunk).all():
						found = True
						break
				if not found:
					break
			state2, reward2, done, infos = env.step(random_act, 1)
			RL.store_transition(state, action1, reward1 - reward2)
			state = state2
			reward1 = reward2 = 0

		# env.render(mode=None)
		# print("state vec =", state)

		# If the game isn't over, change the current player
		if not done:
			user = 0 if user == 1 else 1
		else:
			ep_rs_sum = sum(RL.ep_rs)

			if 'running_reward' not in globals():
				running_reward = ep_rs_sum
			else:
				running_reward = running_reward * 0.8 + ep_rs_sum * 0.2
			if running_reward >= DISPLAY_REWARD_THRESHOLD:
				RENDER = True     # rendering

			if i_episode % 100 == 0:
				rr = int(running_reward)
				print(i_episode, "running reward:", "\x1b[32m" if rr >= 0 else "\x1b[31m", rr, "\x1b[0m")	#, "lr =", RL.lr)
				# RL.set_learning_rate(i_episode)

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
