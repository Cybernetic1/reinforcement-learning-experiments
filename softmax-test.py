import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

N = 10						# dimension of logits
sm = nn.Softmax(dim=0)		# A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
xs = torch.randn(N)		# Returns random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).
# xs = torch.tensor([-1., -1., -1., -1., -1., -1., -1., -1., -1., 9.])
print("xs =", xs)
output1 = sm(xs)
print("output =", output1)
print("sum =", torch.sum(output1))
mean = (1.0 / N) / 2
ys = torch.sub(output1, torch.full((N,), mean))
zs = torch.mul(ys, torch.full((N,), 10.))		# 10.0 = magnifying factor
output2 = sm(zs)
ys = torch.sub(output2, torch.full((N,), mean))
zs = torch.mul(ys, torch.full((N,), 10.))
output3 = sm(zs)
ys = torch.sub(output3, torch.full((N,), mean))
zs = torch.mul(ys, torch.full((N,), 10.))
output4 = sm(zs)
print("iterated output =", output4)

import matplotlib as mpl
import matplotlib.pyplot as plt
# fig, ax = plt.subplots()	# Create a figure containing a single axes
# ax.bar(range(N), output1, width=0.2)
# ax.bar([x-0.2 for x in range(N)], output2, width=0.2)
# ax.bar([x-0.4 for x in range(N)], output3, width=0.2)
# ax.bar([x-0.6 for x in range(N)], output4, width=0.2)
# plt.show()

fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5)	# 1 row 5 columns
fig.set_size_inches(20,5)		# size in inches
ax0.bar(range(N), xs, color="red")
ax0.set_title("Random Logits")

ax1.bar(range(N), output1)
ax1.set_ylim(0, 1.0)
ax1.set_title("Softmax")

ax2.bar(range(N), output2)
ax2.set_ylim(0, 1.0)
ax2.set_title("Softmax²")

ax3.bar(range(N), output3)
ax3.set_ylim(0, 1.0)
ax3.set_title("Softmax³")

ax4.bar(range(N), output4)
ax4.set_ylim(0, 1.0)
ax4.set_title("Softmax⁴")

num = input("file number =?")
plt.savefig("softmax-test" + num + ".png")
plt.show()
