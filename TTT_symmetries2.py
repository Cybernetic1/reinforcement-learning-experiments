from eqPairs import eqPairs

# New method is to build dictionary (s,a) --> class number
Qdict = {}
for i, cls in enumerate(eqPairs):
	for (s,a) in cls:
		Qdict[(s,a)] = i
print("Qdict =", Qdict)

from reachables import reachables
Qdict_s = {}
for s in reachables:
	indices = [0] * 9
	for a in range(0,9):
		indices[a] = Qdict[(s,a)]
	Qdict_s[s] = indices
print("Qdict_s =", Qdict_s)

import pickle
ans = input("\nWrite to Qdicts.pk? [y/N]")
if ans == 'Y' or ans == 'y':
	with open("Qdicts.pk", 'wb') as f1:
		pickle.dump(Qdict, f1)
		pickle.dump(Qdict_s, f1)
	f1.close()
