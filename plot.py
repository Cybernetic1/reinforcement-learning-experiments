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
