import sys
import matplotlib.pyplot as plt

fname = "TTT-test-results.txt"
if len(sys.argv) == 2:
	fname = sys.argv[1]
f = open(fname, 'r')

data = []
for line in f:
	try:
		head, tail = line.split('reward: ')
	except:
		continue
	if tail[0] == '\x1b':
		data.append(int(tail.split(' ')[1]))
	else:
		data.append(int(tail))

print("size=", len(data))

plt.ylabel('reward')
plt.plot(data)
plt.show()

# Old plot:
"""
plt.plot(vt)    # plot the episode vt
plt.xlabel('episode steps')
plt.ylabel('normalized state-action value')
plt.show()
"""

# New plot:
"""
window = int(episodes/20)

fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9,9]);
rolling_mean = pd.Series(policy.reward_history).rolling(window).mean()
std = pd.Series(policy.reward_history).rolling(window).std()
ax1.plot(rolling_mean)
ax1.fill_between(range(len(policy.reward_history)),rolling_mean-std, rolling_mean+std, color='orange', alpha=0.2)
ax1.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
ax1.set_xlabel('Episode'); ax1.set_ylabel('Episode Length')

ax2.plot(policy.reward_history)
ax2.set_title('Episode Length')
ax2.set_xlabel('Episode'); ax2.set_ylabel('Episode Length')

fig.tight_layout(pad=2)
plt.show()
fig.savefig('results-2.png')
"""
