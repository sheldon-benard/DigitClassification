import os
import numpy as np

def getMax(l):
	maxElem = max(l)
	indicies = []
	for i in range(len(l)):
		if maxElem == l[i]:
			indicies.append(i)

	random = np.random.randint(len(indicies))
	return indicies[random]

files = []

for file in os.listdir("../../output"):
    if file.startswith("final"):
    	files.append(file)


fps = []

for file in files:
	fname = "../../output/" + file
	fp = open(fname, "r")
	fps.append(fp)


with open("../../output/test_predictions.csv", 'w') as f:
		ID = 0
		f.write("Id,Label\n")

		loop = True

		while loop:
			count = [0,0,0,0,0,0,0,0,0,0]
			for fp in fps:
				line = fp.readline()
				if not line:
					loop = False
					break
				elif "Label" in line:
					continue
				else:
					line = line.strip()
					num = int(line.split(",")[-1])
					count[num] += 1
			if sum(count) == 0:
				continue #skip first line

			prediction = getMax(count)
			f.write(str(ID) + "," + str(prediction) + "\n")
			ID += 1

