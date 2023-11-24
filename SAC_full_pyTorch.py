"""
Questions：
* SAC 输出是 概率分布 还是 概率本身？ 是前者。
* 而 reparameterization trick 又是否可以避免？ Doesn't matter.
如果是概率分布，则没有了 reparam 的问题？
最重要问题是： 如果是概率分布，在逻辑下是否仍然可行？
但我打算用 Transformer 输出的其实就是 distribution！

而 Reparameterization 的目的是：
1）为了可以计算 随机变量的 gradient
2）减少 variance

Stochastic sampling network
Or the NN outputs the state-conditioned distribution directly
Then the distribution can be sampled
问题是为什么输出才用 reparameterization trick？


TO-DO:
* transfer functions from SAC.py to here
* action space seems different

Fully-connected version, where state vector is a 3 x 3 = 9-vector

Refer to net_config() below for the current network topology and # of weights info.

For example: (9 inputs)-16-16-16-16-(9 outputs)
Total num of weights = 9 * 16 * 2 + 16 * 16 * 3 = 1056
We want num of weights to be close to that of symNN = 1080

============================================================
SAC = soft actor-critic, Reinforcement Learning.  Adapted from:
https://github.com/quantumiracle/Popular-RL-Algorithms

Using:
PyTorch: 1.9.1+cpu
gym: 0.8.0
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.distributions import Normal

# reproducible
np.random.seed(7)
torch.manual_seed(7)
device = torch.device("cpu")

class ReplayBuffer:
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []
		self.position = 0

	def push(self, state, action, reward, next_state, done):
		if len(self.buffer) < self.capacity:
			self.buffer.append(None)
		self.buffer[self.position] = (state, action, reward, next_state, done)
		self.position = (self.position + 1) % self.capacity

	def sum_R(self):
		s = 0
		for d in self.buffer:
			s += d[2]
		return s

	def sample(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
		# print("sampled state=", state)
		# print("sampled action=", action)
		'''
		the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
		zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
		the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
		np.stack((1,2)) => array([1, 2])
		'''
		return state, action, reward, next_state, done

	def __len__(self):
		return len(self.buffer)


class ValueNetwork(nn.Module):
	def __init__(self, state_dim, hidden_dim, activation=F.relu, init_w=3e-3):
		super(ValueNetwork, self).__init__()

		self.linear1 = nn.Linear(state_dim, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, hidden_dim)
		self.linear3 = nn.Linear(hidden_dim, 1)
		# weights initialization
		self.linear3.weight.data.uniform_(-init_w, init_w)
		self.linear3.bias.data.uniform_(-init_w, init_w)

		self.activation = activation

	def forward(self, state):
		x = self.activation(self.linear1(state))
		x = self.activation(self.linear2(x))
		x = self.linear3(x)
		return x


class SoftQNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_size, activation=F.relu, init_w=3e-3):
		super(SoftQNetwork, self).__init__()

		num_actions = 1		# this overrides because output is actually just 1 action
		self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, 1)

		self.linear3.weight.data.uniform_(-init_w, init_w)
		self.linear3.bias.data.uniform_(-init_w, init_w)

		self.activation = activation

	def forward(self, state, action):
		# print("state, action：", state.shape, action.shape)
		# print("action =", action)
		x = torch.cat([state, action[..., None]], dim=-1) # the dim 0 is number of samples
		# print("x：", x.shape)
		x = self.activation(self.linear1(x))
		x = self.activation(self.linear2(x))
		x = self.linear3(x)
		return x


class PolicyNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_size, activation=F.relu, init_w=3e-3, log_std_min=-20, log_std_max=2):
		super(PolicyNetwork, self).__init__()

		self.log_std_min = log_std_min
		self.log_std_max = log_std_max

		self.linear1 = nn.Linear(num_inputs, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, hidden_size)
		self.linear4 = nn.Linear(hidden_size, hidden_size)

		self.mean_linear = nn.Linear(hidden_size, num_actions)
		self.mean_linear.weight.data.uniform_(-init_w, init_w)
		self.mean_linear.bias.data.uniform_(-init_w, init_w)

		self.log_std_linear = nn.Linear(hidden_size, num_actions)
		self.log_std_linear.weight.data.uniform_(-init_w, init_w)
		self.log_std_linear.bias.data.uniform_(-init_w, init_w)

		self.action_range = 2.
		self.num_actions = num_actions
		self.activation = activation


	def forward(self, state):
		x = self.activation(self.linear1(state))
		x = self.activation(self.linear2(x))
		x = self.activation(self.linear3(x))
		x = self.activation(self.linear4(x))

		mean    = (self.mean_linear(x))
		# mean    = F.leaky_relu(self.mean_linear(x))
		log_std = self.log_std_linear(x)
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

		return mean, log_std

	def evaluate(self, state, epsilon=1e-6):
		'''
		generate sampled action with state as input wrt the policy network;
		deterministic evaluation provides better performance according to the original paper;
		'''
		logits, log_std = self.forward(state)
		std = log_std.exp() # no clip in evaluation, clip affects gradients flow

		probs   = torch.softmax(logits, dim=1)
		dist    = Categorical(probs)
		action  = dist.sample()
		""" # **** abandon Reparameterization Trick as it seems non-essential
		normal  = Normal(0, 1)
		z       = normal.sample(probs.shape)
		# TanhNormal distribution as actions; reparameterization trick
		action0 = torch.tanh(probs + std * z.to(device))
		action  = self.action_range * action0 """

		# https://stackoverflow.com/questions/54635355/what-does-log-prob-do
		# print(dist.log_prob(action), torch.log(action_probs[action]))

		# dim-of-action 是 1 还是 9？ 应该是 1
		# 它的值应该是 probs[action] 的值, 但这经过了采样
		# 所以，还是需要 re-parameterization trick？
		# 但 re-param 要求 NN 输出确定的 mean 值，这跟 Transformer 输出的 distro
		# 非常不同。如果想保留 Transformer 输出 distro 的优势，则无法计算 log-prob.
		# The "log" arises from the "log-derivative trick".
		
		log_prob = dist.log_prob(action)
		''' stochastic evaluation '''
		# log_prob = Normal(mean, std).log_prob(mean + std*z.to(device)) - torch.log(1. - action_0.pow(2) + epsilon) -  np.log(self.action_range)
		''' deterministic evaluation '''
		# log_prob = Normal(mean, std).log_prob(mean) - torch.log(1. - torch.tanh(mean).pow(2) + epsilon) -  np.log(self.action_range)
		'''
		both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
		the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
		needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
		'''
		log_prob = log_prob.sum(dim=-1, keepdim=True)
		return action, log_prob		# , z, mean, log_std


	def choose_action(self, state, deterministic):
		""" The actor network's output has 2 components:
		1) either squashed deterministic action a
		   or sampled action a ~ N(μ(s),σ²(s)).
		   The sampling uses the reparameterization trick
		2) log probability that will be needed for calculating H
		"""
		state = torch.FloatTensor(state).unsqueeze(0).to(device)
		logits, log_std = self.forward(state)
		# print("logits=", logits)
		# print("log_std=", log_std)
		std = log_std.exp()

		probs = torch.softmax(logits, dim=1)
		dist   = Categorical(probs)
		action = dist.sample().numpy()[0]
		# *** abandon Reparameterization Trick
		# normal = Normal(0, 1)
		# z      = normal.sample(mean.shape).to(device)
		# action = self.action_range* torch.tanh(mean + std*z)
		# action = torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else action.detach().cpu().numpy()[0]

		# print("chosen action=", action)
		return action


	def sample_action(self,):
		a=torch.FloatTensor(self.num_actions).uniform_(-1, 1)
		return (self.action_range*a).numpy()

class SAC(nn.Module):

	def __init__(
			self,
			action_dim,
			state_dim,
			learning_rate = 3e-4,
			gamma = 0.9 ):
		super(SAC, self).__init__()

		hidden_dim = 512

		self.value_net        = ValueNetwork(state_dim, hidden_dim, activation=F.relu).to(device)
		self.target_value_net = ValueNetwork(state_dim, hidden_dim, activation=F.relu).to(device)

		self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim, activation=F.relu).to(device)
		self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim, activation=F.relu).to(device)
		self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, activation=F.relu).to(device)

		print('(Target) Value Network: ', self.value_net)
		print('Soft Q Network (1,2): ', self.soft_q_net1)
		print('Policy Network: ', self.policy_net)

		for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
			target_param.data.copy_(param.data)

		self.value_criterion  = nn.MSELoss()
		self.soft_q_criterion1 = nn.MSELoss()
		self.soft_q_criterion2 = nn.MSELoss()

		self.value_lr  = learning_rate
		self.soft_q_lr = learning_rate
		self.policy_lr = learning_rate

		self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
		self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=self.soft_q_lr)
		self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=self.soft_q_lr)
		self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)

		self.action_dim = action_dim
		self.state_dim = state_dim

		self.lr = learning_rate
		self.gamma = gamma

		replay_buffer_size = int(1e6)
		self.replay_buffer = ReplayBuffer(replay_buffer_size)


	def update(self, batch_size, reward_scale, gamma=0.99, soft_tau=1e-2):
		alpha = 1.0  # trade-off between exploration (max entropy) and exploitation (max Q)

		state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
		# print('sample (state, action, reward, next state, done):', state, action, reward, next_state, done)

		state      = torch.FloatTensor(state).to(device)
		next_state = torch.FloatTensor(next_state).to(device)
		action     = torch.FloatTensor(action).to(device)
		reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
		done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

		predicted_q_value1 = self.soft_q_net1(state, action)
		predicted_q_value2 = self.soft_q_net2(state, action)
		predicted_value    = self.value_net(state)
		new_action, log_prob = self.policy_net.evaluate(state)

		reward = reward_scale*(reward - reward.mean(dim=0)) /reward.std(dim=0) # normalize with batch mean and std

		# **** Training Q Function
		target_value = self.target_value_net(next_state)
		target_q_value = reward + (1 - done) * gamma * target_value # if done==1, only reward
		q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
		q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())


		self.soft_q_optimizer1.zero_grad()
		q_value_loss1.backward()
		self.soft_q_optimizer1.step()
		self.soft_q_optimizer2.zero_grad()
		q_value_loss2.backward()
		self.soft_q_optimizer2.step()

		# **** Training Value Function
		predicted_new_q_value = torch.min(
			self.soft_q_net1(state, new_action),
			self.soft_q_net2(state, new_action) )
		target_value_func = predicted_new_q_value - alpha * log_prob # for stochastic training, it equals to expectation over action
		value_loss = self.value_criterion(predicted_value, target_value_func.detach())


		self.value_optimizer.zero_grad()
		value_loss.backward()
		self.value_optimizer.step()

		# **** Training Policy Function
		''' implementation 1 '''
		policy_loss = (alpha * log_prob - predicted_new_q_value).mean()
		''' implementation 2 '''
		# policy_loss = (alpha * log_prob - soft_q_net1(state, new_action)).mean()  # Openai Spinning Up implementation
		''' implementation 3 '''
		# policy_loss = (alpha * log_prob - (predicted_new_q_value - predicted_value.detach())).mean() # max Advantage instead of Q to prevent the Q-value drifted high

		''' implementation 4 '''  # version of github/higgsfield
		# log_prob_target=predicted_new_q_value - predicted_value
		# policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
		# mean_lambda=1e-3
		# std_lambda=1e-3
		# mean_loss = mean_lambda * mean.pow(2).mean()
		# std_loss = std_lambda * log_std.pow(2).mean()
		# policy_loss += mean_loss + std_loss


		self.policy_optimizer.zero_grad()
		policy_loss.backward()
		self.policy_optimizer.step()

		# print('value_loss: ', value_loss)
		# print('q loss: ', q_value_loss1, q_value_loss2)
		# print('policy loss: ', policy_loss )


		# **** Soft update the target value net
		for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
			target_param.data.copy_(  # copy data value into target parameters
				target_param.data * (1.0 - soft_tau) + param.data * soft_tau
			)
		return predicted_new_q_value.mean()


	def net_info(self):
		config = "(9)-16-16-16-16-(9)"
		neurons = config.split('-')
		last_n = 9
		total = 0
		for n in neurons[1:-1]:
			n = int(n)
			total += last_n * n
			last_n = n
		total += last_n * 9
		return (config, total)

	def play_random(self, state, action_space):
		# Select an action (0-9) randomly
		# NOTE: random player never chooses occupied squares
		while True:
			action = action_space.sample()
			if state[action] == 0:
				break
		return action

	def clear_data(self):
		# empty episode data
		self.ep_actions = Variable(torch.Tensor())
		self.ep_rewards = []

	def save_net(self, fname):
		torch.save(self.state_dict(), "PyTorch_models/" + fname + ".dict")
		print("Model saved.")

	def load_net(self, fname):
		model = PolicyGradient(9, 9)
		model.load_state_dict(torch.load("PyTorch_models/" + fname + ".dict"))
		model.eval()
		print("Model loaded.")
