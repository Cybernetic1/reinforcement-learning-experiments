import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

sm = nn.Softmax(dim=0)		# A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
xs = torch.randn(9)			# Returns random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).
# xs = torch.tensor([-1., -1., -1., -1., -1., -1., -1., -1., -1., 9.])
print("xs =", xs)
output = sm(xs)
print("output =", output)
print("sum =", torch.sum(output))
ys = torch.sub(output, torch.full((9,), 0.05))
zs = torch.mul(ys, torch.full((9,), 10.))
output = sm(zs)
ys = torch.sub(output, torch.full((9,), 0.05))
zs = torch.mul(ys, torch.full((9,), 10.))
output = sm(zs)
ys = torch.sub(output, torch.full((9,), 0.05))
zs = torch.mul(ys, torch.full((9,), 10.))
output = sm(zs)
print("iterated output =", output)
