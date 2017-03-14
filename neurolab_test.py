import numpy as np
import neurolab as nl
import create_mat
import scipy.io

net = nl.net.newff([[0., 1.]] * (36 * 36), [int(62 * 2.5), 62])
output = None

for i, letter in enumerate(create_mat.LETTERS):
	data = scipy.io.loadmat("data/trainingdata_%d.mat" % letter)
	X = data['X']
	X = np.matrix(X)
	Y = None
	output_matrix = np.matrix(np.zeros((1, 62)))
	output_matrix[(0, i)] = 1.

	for j, image in enumerate(X):
		if output is None:
			output = output_matrix
		else:
			output = np.vstack((output, output_matrix))
	output_matrix[(0, i)] = 1.
	print(X.shape)
	print(output.shape)
	net.train(X, output_matrix, show=15)
	break


