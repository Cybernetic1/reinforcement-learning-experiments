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
env = gym.make('TicTacToe-v1', symbols=[-1, 1], board_size=3, win_size=3)

user = 0
done = False
reward = 0

# Reset the env before playing
state = env.reset()

env.seed(1)     # reproducible, general Policy gradient has high variance

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
	n_actions=env.action_space.n,
	n_features=env.observation_space.shape[0],
	learning_rate=0.02,
	reward_decay=0.99,
	# output_graph=True,
)

for i_episode in range(3000):
	state = env.reset()

	while not done:
		env.render(mode=None)

		action = RL.choose_action(state)

		if user == 0:
			state_, reward, done, infos = env.step(action, -1)
			RL.store_transition(state, action, reward)
		elif user == 1:
			state_, reward, done, infos = env.step(env.action_space.sample(), 1)

		# If the game isn't over, change the current player
		if not done:
			user = 0 if user == 1 else 1
		else:
			if reward == 10:
				print("Draw !")
			elif reward == -20:
				print("Infos : " + str(infos))
				if user == 0:
					print("Random wins ! AI Reward : " + str(reward))
				elif user == 1:
					print("AI wins ! AI Reward : " + str(-reward))
			elif reward == 20:
				if user == 0:
					print("AI wins ! AI Reward : " + str(reward))
				elif user == 1:
					print("Random wins ! AI Reward : " + str(reward))

		state = state_

exit(0)
# ============== Old code from Cartpole example =============

		if done:
			ep_rs_sum = sum(RL.ep_rs)

			if 'running_reward' not in globals():
				running_reward = ep_rs_sum
			else:
				running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
			if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
			print("episode:", i_episode, "  reward:", int(running_reward))

			vt = RL.learn()

			if i_episode == 0:
				plt.plot(vt)    # plot the episode vt
				plt.xlabel('episode steps')
				plt.ylabel('normalized state-action value')
				plt.show()
			break
