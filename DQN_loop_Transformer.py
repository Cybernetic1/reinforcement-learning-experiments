"""
Multi-step experiment with Transformers
RL will output some "intermediate" results that aren't actions.
actions 0-8 = tic-tac-toe actions
actions 9-17 = intermediate thoughts
These will be put into a special area of the "state".
For more explanations see: README-RL-with-autoencoder.md

Primitive Transformer code is taken from:
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html

Using:
PyTorch: 1.9.1+cpu
gym: 0.8.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.distributions import Normal

import random
import numpy as np
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

	def last_reward(self):
		return self.buffer[self.position-1][2]

	def sample(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		state, action, reward, next_state, done = \
			map(np.stack, zip(*batch)) # stack for each element
		'''
		the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
		zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
		the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
		np.stack((1,2)) => array([1, 2])
		'''
		# print("sampled state=", state)
		# print("sampled action=", action)
		return state, action, reward, next_state, done

	def __len__(self):
		return len(self.buffer)

class MultiheadAttention(nn.Module):

	def __init__(self, input_dim, embed_dim, num_heads):
		super().__init__()
		assert embed_dim % num_heads == 0, "Embedding dim must be 0 modulo # of heads."

		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.head_dim = embed_dim // num_heads

		# Stack all weight matrices 1...h together for efficiency
		# Note that in many implementations you see "bias=False" which is optional
		self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
		self.o_proj = nn.Linear(embed_dim, embed_dim)

		self._reset_parameters()

	def _reset_parameters(self):
		# Original Transformer initialization, see PyTorch documentation
		nn.init.xavier_uniform_(self.qkv_proj.weight)
		self.qkv_proj.bias.data.fill_(0)
		nn.init.xavier_uniform_(self.o_proj.weight)
		self.o_proj.bias.data.fill_(0)

	def scaled_dot_product(q, k, v, mask=None):
		d_k = q.size()[-1]
		attn_logits = torch.matmul(q, k.transpose(-2, -1))
		attn_logits = attn_logits / math.sqrt(d_k)
		if mask is not None:
			attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
		attention = F.softmax(attn_logits, dim=-1)
		values = torch.matmul(attention, v)
		return values, attention

	def forward(self, x, mask=None, return_attention=False):
		batch_size, seq_length, _ = x.size()
		if mask is not None:
			mask = expand_mask(mask)
		qkv = self.qkv_proj(x)

		# Separate Q, K, V from linear output
		qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
		qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
		q, k, v = qkv.chunk(3, dim=-1)

		# Determine value outputs
		values, attention = scaled_dot_product(q, k, v, mask=mask)
		values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
		values = values.reshape(batch_size, seq_length, self.embed_dim)
		o = self.o_proj(values)

		if return_attention:
			return o, attention
		else:
			return o

class EncoderBlock(nn.Module):

	def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
		"""
		Inputs:
			input_dim - Dimensionality of the input
			num_heads - Number of heads to use in the attention block
			dim_feedforward - Dimensionality of the hidden layer in the MLP
			dropout - Dropout probability to use in the dropout layers
		"""
		super().__init__()

		# Attention layer
		self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

		# Two-layer MLP
		self.linear_net = nn.Sequential(
			nn.Linear(input_dim, dim_feedforward),
			nn.Dropout(dropout),
			nn.ReLU(inplace=True),
			nn.Linear(dim_feedforward, input_dim)
		)

		# Layers to apply in between the main layers
		self.norm1 = nn.LayerNorm(input_dim)
		self.norm2 = nn.LayerNorm(input_dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, mask=None):
		# Attention part
		attn_out = self.self_attn(x, mask=mask)
		x = x + self.dropout(attn_out)
		x = self.norm1(x)

		# MLP part
		linear_out = self.linear_net(x)
		x = x + self.dropout(linear_out)
		x = self.norm2(x)

		return x

class TransformerEncoder(nn.Module):

	def __init__(self, num_layers, **block_args):
		super().__init__()
		self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

	def forward(self, x, mask=None):
		for l in self.layers:
			x = l(x, mask=mask)
		return x

	def get_attention_maps(self, x, mask=None):
		attention_maps = []
		for l in self.layers:
			_, attn_map = l.self_attn(x, mask=mask, return_attention=True)
			attention_maps.append(attn_map)
			x = l(x)
		return attention_maps

class DQN():

	def __init__(
			self,
			action_dim,
			state_dim,
			learning_rate = 3e-4,
			gamma = 1.0 ):
		super(DQN, self).__init__()

		self.action_dim = action_dim
		self.state_dim = state_dim
		self.lr = learning_rate
		self.gamma = gamma

		self.replay_buffer = ReplayBuffer(int(1e6))

		hidden_dim = 9

		self._build_net()

		self.q_criterion = nn.MSELoss()
		self.q_optimizer = optim.Adam(self.trm.parameters(), lr=self.lr)

	def _build_net(self):

		self.trm = TransformerEncoder(num_layers=1, input_dim=2,dim_feedforward=2*model_dim, num_heads=1, dropout=0.1)

		# W is a 3x9 matrix, to convert 3-vector to 9-vector probability distribution:
		self.W = Variable(torch.randn(2, 9), requires_grad=True)
		self.softmax = nn.Softmax(dim=0)

	def forward(self, x):
		# input dim = n_features = 9 x 2 x 2 = 36
		# First we need to split the input into 18 parts:
		# print("x =", x)
		xs = torch.stack(torch.split(x, 2, 1), 1)
		# print("xs =", xs)
		# There is a question of how these are stacked, 9x3 or 3x9?
		# it has to conform with Transformer's d_model = 3
		ys = self.trm(xs) # no need to split results, already in 9x3 chunks
		# print("ys =", ys)
		# it seems that only the last 3-dim vector is useful
		u = torch.matmul( ys.select(1, 8), self.W )
		# *** sum the probability distributions together:
		# z = torch.stack(zs, dim=1)
		# u = torch.sum(z, dim=1)
		# v = self.softmax(u)
		# print("v =", v)
		return u

	def choose_action(self, state, deterministic=True):
		state = torch.FloatTensor(state).unsqueeze(0).to(device)
		logits = self.symnet(state)
		probs  = torch.softmax(logits, dim=1)
		dist   = Categorical(probs)
		action = dist.sample().numpy()[0]

		# print("chosen action=", action)
		return action

	def update(self, batch_size, reward_scale, gamma=1.0):
		# alpha = 1.0  # trade-off between exploration (max entropy) and exploitation (max Q);  not used now

		state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
		# print('sample (state, action, reward, next state, done):', state, action, reward, next_state, done)

		state      = torch.FloatTensor(state).to(device)
		next_state = torch.FloatTensor(next_state).to(device)
		action     = torch.LongTensor(action).to(device)
		reward     = torch.FloatTensor(reward).to(device) # .to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
		done       = torch.BoolTensor(done).to(device)

		logits = self.symnet(state)
		next_logits = self.symnet(next_state)

		# **** Train deep Q function, this is just Bellman equation:
		# DQN(st,at) += η [ R + γ max_a DQN(s_t+1,a) - DQN(st,at) ]
		# DQN[s, action] += self.lr *( reward + self.gamma * np.max(DQN[next_state, :]) - DQN[s, action] )
		# max 是做不到的，但似乎也可以做到。 DQN 输出的是 probs.
		# probs 和 Q 有什么关系？  Q 的 Boltzmann 是 probs (SAC 的做法).
		# This implies that Q = logits.
		# logits[at] += self.lr *( reward + self.gamma * np.max(logits[next_state, next_a]) - logits[at] )
		q = logits[range(logits.shape[0]), action]
		# maxq = torch.softmax(next_logits, 1, keepdim=False).values
		softmaxQ = torch.log(torch.sum(torch.exp(next_logits), 1))
		# print("softmaxQ:", softmaxQ.shape)
		# q = q + self.lr *( reward + self.gamma * m - q )
		# torch.where: if condition then arg2 else arg3
		target_q = torch.where(done, reward, reward + self.gamma * softmaxQ)
		# print("q, target_q:", q.shape, target_q.shape)
		q_loss = self.q_criterion(q, target_q.detach())

		self.q_optimizer.zero_grad()
		q_loss.backward()
		self.q_optimizer.step()

		return

	def visualize_q(self, board, memory):
		# convert board vector to state vector
		vec = []
		for i in range(9):
			symbol = board[i]
			vec += [symbol, i-4]
		for i in range(9):
			if memory[i] == 1:
				vec += [-2, i-4]
			else:
				vec += [2,0]
		state = torch.FloatTensor(vec).unsqueeze(0).to(device)
		logits = self.symnet(state)
		probs  = torch.softmax(logits, dim=1)
		return probs.squeeze(0)

	def net_info(self):
		config_h = "(2)-9-9"
		config_g = "9-9-(9)"
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
		return (config_h + ':' + config_g, total)

	def play_random(self, state, action_space):
		# Select an action (0-9) randomly
		# NOTE: random player never chooses occupied squares
		empties = [0,1,2,3,4,5,6,7,8]
		# Find and collect all empty squares
		# scan through all 9 propositions, each proposition is a 2-vector
		for i in range(0, 18, 2):
			# 'proposition' is a numpy array[3]
			proposition = state[i : i + 2]
			sym = proposition[0]
			if sym == 1 or sym == -1:
				x = proposition[1]
				j = x + 4
				empties.remove(j)
		# Select an available square randomly
		action = random.sample(empties, 1)[0]
		return action

	def save_net(self, fname):
		torch.save(self.symnet.state_dict(), \
			"PyTorch_models/" + fname + ".dict")
		print("Model saved.")

	def load_net(self, fname):
		self.symnet.load_state_dict(torch.load("PyTorch_models/" + fname + ".dict"))
		self.symnet.eval()
		print("Model loaded.")
