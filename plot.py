import matplotlib.pyplot as plt

f = open("TTT2-test2-results.txt", 'r')

data = []
for line in f:
	try:
		head, tail = line.split('reward: ')
	except:
		continue
	data.append(int(tail))

plt.plot(data)
plt.ylabel('reward')
plt.show()
