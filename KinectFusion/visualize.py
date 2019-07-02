import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt("tsdf.txt");
data = np.reshape(data, data.shape + (1,))
print(data.shape)

data = np.reshape(data, (-1, 128, 128))
print(data.shape)

for idx in range(10):
	layer = data[60 + idx]
	layer[layer > 0.1] = 0
	layer[layer < -0.1] = 0
	layer[layer != 0] = 255

	plt.imshow(layer, cmap='gray', interpolation='nearest')
	plt.show()
