"""
Symmetric NN version.

Network topology: 9-inputs, h = -6-9-, g = -9-9- 
======================================================
This part of code is the reinforcement learning brain,
which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 2.0
gym: 0.8.0
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import keras.backend as K
from keras.layers import Dense, Embedding, Lambda, Reshape, Input, Concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam

# reproducible
np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient:
	def __init__(
			self,
			n_actions,
			n_features,
			# learning_rate=0.01,
			reward_decay=0.99,
			output_graph=True,
	):
		self.n_actions = n_actions
		self.n_features = n_features
		self.gamma = reward_decay

		# learning rate = A exp(-k i)
		# when i = 1, rate = 0.01
		# when i = 100000, rate = 0.001
		# self.A = 0.003
		# self.k = 1.00000
		self.lr = 0.001

		self.ep_obs, self.ep_as, self.ep_rs = [], [], []

		self._build_net()

		self.sess = tf.Session()

		if output_graph:
			# $ tensorboard --logdir=logs
			# http://0.0.0.0:6006/
			# tf.train.SummaryWriter soon be deprecated, use following
			tf.summary.FileWriter("logs/", self.sess.graph)

		self.sess.run(tf.global_variables_initializer())

	def _build_net(self):
		# with tf.name_scope('inputs'):
		self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
		self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
		self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

		# input_txt=Input(tensor=self.tf_obs)
		# print("input_txt shape=", input_txt.shape)
		# print("input_txt dtype=", input_txt.dtype)
		# Embedding(input dim, output dim, ...)
		# x = Embedding(self.n_features, self.n_features * 2, mask_zero=False)(input_txt)
		# x2 = Reshape((3, 9))(x)
		xs = tf.split(self.tf_obs, 9, axis=1)

		"""  *** DEMO CODE ***
		h = Dense(3, activation='tanh')
		ys = []
		for i in range(9):
			ys.append( h(xs[i]) )
		y = Keras.stack(ys, axis=1)
		Adder = Lambda(lambda x: Keras.sum(x, axis=1))
		y = Adder(y)
		g = Dense(3)
		output = g(y)
		"""

		shared_layer1 = Dense(8, input_shape=(None, 3), activation='tanh')
		xs = tf.split(self.tf_obs, 9, axis=1)		# split tensor into 9 parts, each part is 3-dims wide. ie, shape = [None, 3]
		# output shape = [None, 6]
		ys = []
		for i in range(9):							# repeat the 1st layer 9 times
			ys.append( shared_layer1(xs[i]) )
		# print("y0 shape=", ys[0].shape)
		shared_layer2 = Dense(9, input_shape=(None, 8), activation='tanh')		# output shape = [None, 9]
		zs = []
		for i in range(9):							# repeat the 2nd layer 9 times
			zs.append( shared_layer2(ys[i]) )
		# print("z0 shape=", zs[0].shape)
		z = K.stack(zs, axis=1)						# output zs = 9 * [None, 9].
		# print("z shape after stack=", z.shape)
		Adder = Lambda(lambda x: K.sum(x, axis=1))	# whatever this is, it means summing over the 9 dimensions
		z = Adder(z)
		# print("z shape after Adder=", z.shape)
		z2 = Dense(self.n_actions + 3, input_shape=(None, self.n_actions), activation='tanh')(z)				# input shape = [None, 9]
		all_act = Dense(self.n_actions, input_shape=(None, self.n_actions + 3), activation=None)(z2)			# [None, 9] again

		# Total number of (independent) weights = 3 * 6 + 6 * 9 + 9 * 9 + 9 * 9 = 18 + 54 + 81 + 81 = 234.
		# Alternatively if: 3 * 6 + 6 * 8 + 8 * 9 + 9 * 9 = 18 + 48 + 72 + 81 = 90 + 40 + 89 = 130 + 89 = 219.
		# Total number of weights (including repeated ones) = 3*6*9 + 6*9*9 + 9*9 + 9*9 = 162 + 486 + 81 + 81 = 810.

		self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

		with tf.name_scope('loss'):
			# to maximize total reward (log_p * R) is to minimize -(log_p * R), and TF only has minimize(loss)
			print("logits shape=", all_act.shape)			# (None, 9)
			print("labels shape=", self.tf_acts.shape)		# (None, )  1-hot
			neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
			# or in this way:
			# neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
			loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

		with tf.name_scope('train'):
			self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

	def choose_action(self, observation):
		prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
		# print("probs=", prob_weights)
		p = prob_weights.reshape(-1)
		# print("p=", p)
		action = np.random.choice(range(prob_weights.shape[1]), p=p)  # select action w.r.t the actions prob
		return action

	def store_transition(self, s, a, r):		# state, action, reward
		self.ep_obs.append(s)
		self.ep_as.append(a)
		self.ep_rs.append(r)

	def learn(self):
		# discount and normalize episode reward
		discounted_ep_rs_norm = self._discount_and_norm_rewards()

		# obs = np.vstack(self.ep_obs)
		# print("*shape obs=", obs.shape)
		# print("*dtype obs=", obs.dtype)
		# acts = np.array(self.ep_as)
		# print("*shape acts=", acts.shape)
		# print("*dtype acts=", acts.dtype)
		# acts2 = np.vstack(self.ep_as)
		# print("*shape acts2=", acts2.shape)

		# train on episode
		self.sess.run(self.train_op, feed_dict={
			 self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
			 self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
			 self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
		})

		self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
		return discounted_ep_rs_norm

	def _discount_and_norm_rewards(self):
		# discount episode rewards
		# print("ep_rs=", self.ep_rs)
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

	# Not used, because Adam takes care of learning rate
	def set_learning_rate(self, i):
		self.lr = self.A * np.exp(- self.k * i)
