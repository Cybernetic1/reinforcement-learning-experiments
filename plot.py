import sys
import matplotlib.pyplot as plt
import pandas as pd		# for calculating rolling mean

import glob
import os
files = glob.glob("results/results.*.txt")
files.sort(key=os.path.getmtime)
for i, fname in enumerate(files):
	if i % 2:
		print(end="\x1b[32m")
	else:
		print(end="\x1b[0m")
	print("%2d. %s (%d)" %(i, fname[16:-4], os.stat(fname).st_size))
print(end="\x1b[0m")

s = input("Enter one or two file number (eg. 1,2): ").split(',')
c = int(s[0])
if len(s) == 2:
	c2 = int(s[1])
else:
	c2 = -1

"""
if len(sys.argv) == 2:
	fname = sys.argv[1]
f = open(fname, 'r')
"""

f = open(files[c], 'r')
tag = ' '.join( files[c][16:-4].split('.', 2) )
data = []

"""		**** Old Format ****
for line in f:
	if line[0] == '#':		# comments
		continue
	try:
		head, tail = line.split('reward: ')
	except:
		continue
	if tail[0] == '\x1b':
		data.append(int(tail.split(' ')[1]))
	else:
		data.append(int(tail))
"""

for line in f:
	if line[0] == '#':		# comments
		continue
	try:
		head, tail = line.split(' ')
	except:
		continue
	data.append(float(tail))

print("size=", len(data))

if c2 >= 0:
	f2 = open(files[c2], 'r')
	tag2 = ' '.join( files[c2][16:-4].split('.', 2) )
	data2 = []

	for line in f2:
		if line[0] == '#':		# comments
			continue
		try:
			head, tail = line.split(' ')
		except:
			continue
		data2.append(float(tail))

	print("size2=", len(data2))

""" *** chop to about same size, +100
if len(data) > len(data2):
	data = data[: len(data2) + 100]
else:
	data2 = data2[: len(data) + 100]
"""

"""  *** Old Plot ***
plt.xlabel('episodes')
plt.ylabel('reward')
plt.plot(data)
plt.show()
"""

window = int(len(data)/20)

fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9,9])

ax1.set_title('Rewards Moving Average ({}-episode window)'.format(window))
ax1.set_xlabel('Episode'); ax1.set_ylabel('Reward')

rolling_mean = pd.Series(data).rolling(window).mean()
std = pd.Series(data).rolling(window).std()
ax1.plot(rolling_mean, color='red')
ax1.fill_between(range(len(data)),rolling_mean - std, rolling_mean + std, color='red', alpha=0.2, label='_nolegend_')

if c2 >=0:
	rolling_mean2 = pd.Series(data2).rolling(window).mean()
	std2 = pd.Series(data2).rolling(window).std()
	ax1.plot(rolling_mean2, color='blue')
	ax1.fill_between(range(len(data2)),rolling_mean2 - std2, rolling_mean2 + std2, color='blue', alpha=0.2, label='_nolegend_')

	ax1.legend([tag, tag2])
else:
	ax1.legend([tag])

# -----------------

ax2.set_title('Rewards')
ax2.set_xlabel('Episode'); ax2.set_ylabel('Reward')

ax2.plot(data, color='red', alpha=0.6)
if c2 >= 0:
	ax2.plot(data2, color='blue', alpha=0.5)
	ax2.legend([tag, tag2])
else:
	ax2.legend([tag])

fig.tight_layout(pad=2)
plt.show()
fig.savefig('results.png')
