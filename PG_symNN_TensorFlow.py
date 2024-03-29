"""
Symmetric NN version.

Refer to net_config() below for the current network topology and total number of weights info.

For example: 9 x 3 inputs, h = (3)-8-(9), g = (9)-12-(9)
Total # of weights: (3*8 + 8*9) *9 + 9*12 + 12*9 = 1080
Duplicate weights are counted because they are updated multiple times.

======================================================
Policy Gradient, Reinforcement Learning.  Adapted from:
Morvan Zhou's tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 2.0 (as 1.0)
gym: 0.8.0
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import keras.backend as K
from keras.layers import Dense, Embedding, Lambda, Reshape, Input, Concatenate
# from keras.models import Model, load_model
# from keras.optimizers import Adam

# reproducible
numpy_seed = 1
np.random.seed(numpy_seed)
tensorflow_seed = 1
tf.set_random_seed(tensorflow_seed)

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
		self.gamma = gamma

		# learning rate = A exp(-k i)
		# when i = 1, rate = 0.01
		# when i = 100000, rate = 0.001
		# self.A = 0.003
		# self.k = 1.00000
		# self.lr = learning_rate
		# This makes the learning rate adjustable during training:
		self.lr = tf.Variable(learning_rate, trainable=False)

		self.ep_obs, self.ep_actions, self.ep_rewards = [], [], []

		self._build_net()

		self.sess = tf.Session()

		# **** Prepare for saving model:
		# self.checkpoint_path = "training/checkpoint1"
		# self.callback = tf.keras.callbacks.ModelCheckpoint( \
		#	filepath = self.checkpoint_path, \
		#	save_weights_only=True, verbose=1 )

		if output_graph:
			tf.summary.FileWriter("logs/", self.sess.graph)

		self.sess.run(tf.global_variables_initializer())

	def net_info(self):
		config_h = "(3)-8-9"
		config_g = "9-12-(9)"
		total = 0
		neurons = config_h.split('-')
		last_n = 3
		for n in neurons[1:]:
			n = int(n)
			total += last_n * n
			last_n = n
		total *= 9

		neurons = config_g.split('-')
		for n in neurons[1:-1]:
			n = int(n)
			total += last_n * n
			last_n = n
		total += last_n * 9
		return (config_h + 'x' + config_g, total)

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

		"""  *** DEMO CODE, for slides presentation only ***
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

		xs = tf.split(self.tf_obs, 9, axis=1)		# split tensor into 9 parts, each part is 3-dims wide. ie, shape = [None, 3]
		# output shape = [None, 6]
		ys = []
		shared_layer1 = Dense(8, input_shape=(None, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01))
		for i in range(9):							# repeat the 1st layer 9 times
			ys.append( shared_layer1(xs[i]) )
		# print("y0 shape=", ys[0].shape)
		shared_layer2 = Dense(9, input_shape=(None, 8), activation=tf.keras.layers.LeakyReLU(alpha=0.01))		# output shape = [None, 9]
		zs = []
		for i in range(9):							# repeat the 2nd layer 9 times
			zs.append( shared_layer2(ys[i]) )
		# print("z0 shape=", zs[0].shape)
		z = K.stack(zs, axis=1)						# output zs = 9 * [None, 9].
		# print("z shape after stack=", z.shape)
		Adder = Lambda(lambda x: K.sum(x, axis=1))	# whatever this is, it means summing over the 9 dimensions
		z = Adder(z)
		# print("z shape after Adder=", z.shape)
		z2 = Dense(self.n_actions + 3, input_shape=(None, self.n_actions), activation=tf.keras.layers.LeakyReLU(alpha=0.01))(z)				# input shape = [None, 9]
		all_act = Dense(self.n_actions, input_shape=(None, self.n_actions + 3), activation=None)(z2)			# [None, 9] again

		self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

		with tf.name_scope('loss'):
			# to maximize total reward (log_p * R) is to minimize -(log_p * R), and TF only has minimize(loss)
			# print("logits shape=", all_act.shape)			# (None, 9)
			# print("labels shape=", self.tf_acts.shape)		# (None, )  1-hot
			neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
			# or in this way:
			# neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
			loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

		with tf.name_scope('train'):
			self.opt = tf.train.AdamOptimizer(self.lr)
			self.train_op = self.opt.minimize(loss)

	def choose_action(self, observation):
		prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
		# print("probs=", prob_weights)
		p = prob_weights.reshape(-1)
		# print("p=", p)
		action = np.random.choice(range(prob_weights.shape[1]), p=p)  # select action w.r.t the actions prob
		return action

	def play_random(self, state, action_space):
		# Select an action (0-9) randomly
		# NOTE: random player never chooses occupied squares
		while True:
			action = action_space.sample()
			x = action % 3
			y = action // 3
			occupied = False
			for i in range(0, 27, 3):		# scan through all 9 propositions, each proposition is a 3-vector
				# 'proposition' is a numpy array[3]
				proposition = state[i : i + 3]
				# print("proposition=",proposition)
				if ([x,y,1] == proposition).all():
					occupied = True
					break
				if ([x,y,-1] == proposition).all():
					occupied = True
					break
			if not occupied:
				break
		return action

	def store_transition(self, s, a, r):		# state, action, reward
		self.ep_obs.append(s)
		self.ep_actions.append(a)
		self.ep_rewards.append(r)

	def learn(self):
		# discount and normalize episode reward
		discounted_ep_rewards_norm = self._discount_and_norm_rewards()

		# obs = np.vstack(self.ep_obs)
		# print("*shape obs=", obs.shape)
		# print("*dtype obs=", obs.dtype)
		# acts = np.array(self.ep_actions)
		# print("*shape acts=", acts.shape)
		# print("*dtype acts=", acts.dtype)
		# acts2 = np.vstack(self.ep_actions)
		# print("*shape acts2=", acts2.shape)

		# train on episode
		self.sess.run(self.train_op, feed_dict={
			self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
			self.tf_acts: np.array(self.ep_actions),  # shape=[None, ]
			self.tf_vt: discounted_ep_rewards_norm,  # shape=[None, ]
			})

		self.ep_obs, self.ep_actions, self.ep_rewards = [], [], []    # empty episode data
		return discounted_ep_rewards_norm

	def _discount_and_norm_rewards(self):
		# discount episode rewards
		# print("ep_rewards=", self.ep_rewards)
		discounted_ep_rewards = np.zeros_like(self.ep_rewards)
		running_add = 0
		for t in reversed(range(0, len(self.ep_rewards))):
			running_add = running_add * self.gamma + self.ep_rewards[t]
			discounted_ep_rewards[t] = running_add

		# normalize episode rewards
		# print("discounted episode rewards=", discounted_ep_rewards)
		# discounted_ep_rewards -= np.mean(discounted_ep_rewards)
		# discounted_ep_rewards /= np.std(discounted_ep_rewards)
		return discounted_ep_rewards

	def clear_data(self):
		# empty episode data
		self.ep_actions = []
		self.ep_rewards = []

	def save_net(self, fname):
		saver = tf.train.Saver()
		save_path = saver.save(self.sess, "TensorFlow_models/" + fname + ".ckpt")
		print("Model saved as: %s" % save_path)

	def load_net(self, fname):
		saver = tf.train.Saver()
		saver.restore(self.sess, "TensorFlow_models/" + fname + ".ckpt")
		print("Model loaded.")

	# Not used, because ADAM takes care of learning rate
	def set_learning_rate(self, lr):
		# self.lr = self.A * np.exp(- self.k * i)
		self.lr.assign(lr)
