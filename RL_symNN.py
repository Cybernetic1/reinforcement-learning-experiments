"""
This is the plain version, where the state vector is a 3 x 3 = 9-vector

Network topology = 9-inputs -9-7-5- 9-outputs

============================================================
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
PyTorch: 1.9.0+cpu
gym: 0.8.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

# reproducible
np.random.seed(1)
torch.manual_seed(1)

class PolicyGradient(nn.Module):
	def __init__(
			self,
			n_actions,
			n_features,
			learning_rate=0.01,
			gamma=0.95,
	):
		super(PolicyGradient, self).__init__()
		self.n_actions = n_actions
		self.n_features = n_features

		self.lr = learning_rate
		self.gamma = gamma

		self.ep_obs, self.ep_as, self.ep_rs = [], [], []

		# Episode policy and reward history
		self.policy_history = Variable(torch.Tensor())
		self.reward_episode = []
		# Overall reward and loss history
		self.reward_history = []				# = ep_rs ?
		self.loss_history = []

		self._build_net()

		self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
		# self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

	def _build_net(self):
		self.l1 = nn.Linear(self.n_features, 9, bias=True)
		self.l2 = nn.Linear(9, 7, bias=True)
		self.l3 = nn.Linear(7, 5, bias=True)
		self.l4 = nn.Linear(5, self.n_actions, bias=False)

		# self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
		# self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
		# self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

		# total number of weights = 9 * 9 + 9 * 7 + 7 * 5 + 5 * 9 = 81 + 63 + 35 + 45 = 224

	def forward(self, x):
		model = torch.nn.Sequential(
			self.l1,
			nn.Dropout(p=0.6),
			nn.ReLU(),
			self.l2,
			nn.ReLU(),
			self.l3,
			nn.ReLU(),
			self.l4,
			nn.Softmax(dim=-1),
			)
		return model(x)

		# use softmax to convert to probability:
		# self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

		#with tf.name_scope('loss'):
		# # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
		# print("logits shape=", all_act.shape)
		# print("labels shape=", self.tf_acts.shape)
		# neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
		# # or in this way:
		# # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
		# loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

	def choose_action(self, state):
		#Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
		state = torch.from_numpy(state).type(torch.FloatTensor)
		state = self(Variable(state))
		c = Categorical(state)
		action = c.sample()

		# Add log probability of our chosen action to our history
		log_probs = c.log_prob(action).unsqueeze(0)
		if self.policy_history.dim() != 0:
			# print("log probs:", log_probs)
			self.policy_history = torch.cat([self.policy_history, log_probs])
		else:
			self.policy_history = (log_probs)
		return action

	# def choose_action(self, observation):
		# prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
		# action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
		# return action

	def store_transition(self, s, a, r):		# state, action, reward
		self.ep_obs.append(s)
		self.ep_as.append(a)
		self.ep_rs.append(r)

	def learn(self):
		R = 0
		rewards = []

		# Discount future rewards back to the present using gamma
		# print("\nLength of reward episode:", len(self.ep_rs)) 
		for r in self.ep_rs[::-1]:			# [::-1] reverses a list
			R = r + self.gamma * R
			rewards.insert(0, R)

		# Scale rewards
		rewards = torch.FloatTensor(rewards)
		rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

		# Calculate loss
		# print("policy history:", self.policy_history)
		# print("rewards:", rewards)
		loss = (torch.sum(torch.mul(self.policy_history, Variable(rewards)).mul(-1), -1))

		# Update network weights
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		#Save and intialize episode history counters
		self.loss_history.append(loss.item())
		self.reward_history.append(np.sum(self.ep_rs))
		self.policy_history = Variable(torch.Tensor())
		self.reward_episode= []

		self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
		return rewards		# == discounted_ep_rs_norm

	# def learn(self):
		# # discount and normalize episode reward
		# discounted_ep_rs_norm = self._discount_and_norm_rewards()

		# # obs = np.vstack(self.ep_obs)
		# # print("*shape obs=", obs.shape)
		# # print("*dtype obs=", obs.dtype)
		# # acts = np.array(self.ep_as)
		# # print("*shape acts=", acts.shape)
		# # print("*dtype acts=", acts.dtype)

		# # train on episode
		# self.sess.run(self.train_op, feed_dict={
			 # self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
			 # self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
			 # self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
		# })

	def _discount_and_norm_rewards(self):
		# discount episode rewards
		discounted_ep_rs = np.zeros_like(self.ep_rs)
		running_add = 0
		for t in reversed(range(0, len(self.ep_rs))):
			running_add = running_add * self.gamma + self.ep_rs[t]
			discounted_ep_rs[t] = running_add

		# normalize episode rewards
		# print("discounted episode rewards=", discounted_ep_rs)
		discounted_ep_rs -= np.mean(discounted_ep_rs)
		discounted_ep_rs /= np.std(discounted_ep_rs)
		return discounted_ep_rs