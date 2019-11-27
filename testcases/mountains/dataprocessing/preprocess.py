import numpy as np
import os

FILES = os.listdir("./npys")
try:
		os.makedirs("./dataset")
except:
		pass

RESOLUTION = 512

def invalid(data, percentage):
		return np.count_nonzero(data) < data.size*percentage or (data < 0).any()

def rotate(data):
	return np.rot90(data)

def reflect(data):
	return np.flip(data)

for ind, i in enumerate(FILES):
		print("Processing", i, "{}/{}".format(ind + 1, len(FILES)))
		data = np.load(os.path.join("./npys/", i))
		_x, _y = data.shape
		w = len(str(_x))
		for y in range(0, _y - RESOLUTION, RESOLUTION):
				for x in range(0, _x - RESOLUTION, RESOLUTION):
						item = data[y:y + RESOLUTION, x:x + RESOLUTION]
						if invalid(item, 0.5): continue
						s = 0
						r = 0
						#for s in range(2):
						#	for r in range(4):
						np.save(os.path.join("./dataset/", i[:-4] + "_{:0{width}}_{:0{width}}_{}{}_{}.npy".format(x, y, s, r, RESOLUTION, width=w)), item)
						#		item = rotate(item)
						#	item = reflect(item)
print("Done")	
