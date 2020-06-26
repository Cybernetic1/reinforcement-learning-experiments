"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
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
from keras.layers import Dense, Embedding, Lambda, Reshape, Input
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
			learning_rate=0.01,
			reward_decay=0.95,
			output_graph=True,
	):
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.gamma = reward_decay

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
		self.tf_obs = tf.placeholder(tf.int32, [None, self.n_features], name="observations")
		self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
		self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

		input_txt=Input(tensor=self.tf_obs)
		print("input_txt shape=", input_txt.shape)
		print("datatype=", input_txt.dtype)
		# Embedding(input dim, output dim, ...)
		x = Embedding(self.n_features, self.n_features * 2, mask_zero=False)(input_txt)
		x2 = Reshape((self.n_features, 3, 3))(x)
		x = Dense(self.n_features, activation='tanh')(x2)
		Adder = Lambda(lambda x: K.sum(x, axis=1))
		x = Adder(x)
		print("x shape=", x.shape)
		encoded = Dense(self.n_actions)(x)
		# summer = Model(input_txt, self.encoded)
		# adam = Adam(lr=1e-4, epsilon=1e-3)
		# summer.compile(optimizer=adam, loss='mae')

		"""
		# fc1
		layer1 = tf.layers.dense(
			inputs=self.tf_obs,
			units=9,
			activation=tf.nn.tanh,  # tanh activation
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.1),
			name='fc1'
		)
		# fc2
		layer2 = tf.layers.dense(
			inputs=layer1,
			units=7,
			activation=tf.nn.tanh,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.1),
			name='fc2'
		)
		# fc3
		layer3 = tf.layers.dense(
			inputs=layer2,
			units=5,
			activation=tf.nn.tanh,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.1),
			name='fc3'
		)
		# fc4
		all_act = tf.layers.dense(
			inputs=layer3,
			units=self.n_actions,
			activation=None,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.1),
			name='fc4'
		)
		"""

		self.all_act_prob = tf.nn.softmax(encoded, name='act_prob')  # use softmax to convert to probability

		with tf.name_scope('loss'):
			# to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
			neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=encoded, labels=self.tf_acts)   # this is negative log of chosen action
			# or in this way:
			# neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
			loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

		with tf.name_scope('train'):
			self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

	def choose_action(self, observation):
		prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
		action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
		return action

	def store_transition(self, s, a, r):		# state, action, reward
		self.ep_obs.append(s)
		self.ep_as.append(a)
		self.ep_rs.append(r)

	def learn(self):
		# discount and normalize episode reward
		discounted_ep_rs_norm = self._discount_and_norm_rewards()

		input_txt = np.vstack(self.ep_obs)
		print("*shape input_txt=", input_txt.shape)
		print("*dtype input_txt=", input_txt.dtype)

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

