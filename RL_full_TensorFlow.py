"""
This is the 'plain' version, where the state vector is a 3 x 3 = 9-vector

Network is automatically generated from the string 'self.topology'

For example:  topology = (9 inputs)-12-12-12-12-12-12-12-(9 outputs)
Total # of weights = 9 * 12 * 2 + 12 * 12 * 6 = 1080

Another example: (9 inputs)-16-16-16-16-(9 outputs)
Total # of weights = 9 * 16 * 2 + 16 * 16 * 3 = 1056

We want # of weights to be close to that of symNN = 1080

=====================================================================
Policy Gradient, Reinforcement Learning.  Adapted from:
Morvan Zhou's tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 2.0 (as 1.0)
gym: 0.8.0
"""

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

# reproducible
np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient:
	def __init__(
			self,
			n_actions,
			n_features,
			learning_rate,
			gamma,
			output_graph=False,
	):
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.gamma = gamma

		self.ep_obs, self.ep_as, self.ep_rs = [], [], []

		# self.topology = "(9)-12-12-12-12-12-12-12-(9)"
		self.topology = "(9)-16-16-16-16-(9)"
		self._build_net()

		self.sess = tf.Session()

		if output_graph:
			# $ tensorboard --logdir=logs
			# http://0.0.0.0:6006/
			tf.summary.FileWriter("logs/", self.sess.graph)

		self.sess.run(tf.global_variables_initializer())

	def net_info(self):
		neurons = self.topology.split('-')
		last_n = 9
		total = 0
		for n in neurons[1:-1]:
			n = int(n)
			total += last_n * n
			last_n = n
		total += last_n * 9
		return (self.topology, total)

	def _build_net(self):
		with tf.name_scope('inputs'):
			self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
			self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
			self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

		neurons = self.topology.split('-')
		layer = []
		for i, n in enumerate(neurons[1:-1]):
			n = int(n)
			layer.append(tf.layers.dense(
				inputs = self.tf_obs if i == 0 else layer[i - 1],
				units = n,
				activation = tf.nn.tanh,
				kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),
				bias_initializer = tf.constant_initializer(0.1),
				name='fc' + str(i) ))

		# Last layer
		all_act = tf.layers.dense(
			inputs = layer[len(neurons) - 3],	# layers start from 0
			units = self.n_actions,
			activation = None,
			kernel_initializer = tf.random_normal_initializer( mean=0, stddev=0.3 ),
			bias_initializer = tf.constant_initializer(0.1),
			name='fc-last' )

		self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

		with tf.name_scope('loss'):
			# to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
			print("logits = all_act, shape=", all_act.shape)
			print("labels shape=", self.tf_acts.shape)
			neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
			# or in this way:
			# neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
			loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward-guided loss

		with tf.name_scope('train'):
			self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

	# **** Input is just one state
	def choose_action(self, observation):
		prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})

		action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
		return action

	def play_random(self, state, action_space):
		# Select an action (0-9) randomly
		# NOTE: random player never chooses occupied squares
		while True:
			action = action_space.sample()
			if state[action] == 0:
				break
		return action

	def store_transition(self, s, a, r):		# state, action, reward
		self.ep_obs.append(s)
		self.ep_as.append(a)
		self.ep_rs.append(r)

	# **** Train on an entire episode = 1 game
	def learn(self):
		# discount and normalize episode reward
		discounted_ep_rs_norm = self._discount_and_norm_rewards()

		# obs = np.vstack(self.ep_obs)
		# print("*shape obs=", obs.shape)
		# print("*dtype obs=", obs.dtype)
		# acts = np.array(self.ep_as)
		# print("*shape acts=", acts.shape)
		# print("*dtype acts=", acts.dtype)

		# train on one episode
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
		# discounted_ep_rs -= np.mean(discounted_ep_rs)
		# discounted_ep_rs /= np.std(discounted_ep_rs)
		return discounted_ep_rs

	def save_net(self, fname):
		saver = tf.train.Saver()
		save_path = saver.save(self.sess, "TensorFlow_models/" + fname + ".ckpt")
		print("Model saved as: %s" % save_path)

	def load_net(self, fname):
		saver = tf.train.Saver()
		saver.restore(self.sess, "TensorFlow_models/" + fname + ".ckpt")
		print("Model loaded.")
