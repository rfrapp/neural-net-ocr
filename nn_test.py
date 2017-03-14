
import NeuralNetwork, numpy, collections
import scipy.io, scipy.misc
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.mlab as mlab
import create_mat

def display_image(imgdata):
	fig = plt.figure()

	ax0 = fig.add_subplot(221)
	ax0.axis('off')
	ax0.imshow(imgdata, cmap='gray')

	plt.tight_layout()
	plt.show(block=False)
	s = input("Press Enter to close...")
	plt.close()

	return s

data = scipy.io.loadmat("debug.mat")
Theta1 = numpy.matrix(data['iw1'])
print(Theta1)
Theta2 = numpy.matrix(data['iw2'])
input_m = numpy.matrix(data['i'])[1:]
input_m_2d = numpy.reshape(input_m, (36, 36))
y = numpy.matrix(data['y'])
display_image(input_m_2d)

num_inputs = create_mat.COL_SIZE ** 2
hidden_size = int(len(create_mat.LETTERS) * 2.5)
output_size = len(create_mat.LETTERS)

nn = NeuralNetwork.NeuralNetwork(input_size=num_inputs, output_size=output_size)
nn.append_layer(rows=hidden_size, cols=num_inputs + 1,
	            is_output=False, weights=Theta1)
nn.append_layer(rows=output_size, cols=hidden_size + 1, is_output=True,
				weights=Theta2)
nn.add_training_row(input_matrix=input_m, output_matrix=y)

nn.set_input(0)
nn.forward_propagate()
print(nn.activations)
# nn.backward_propagate(learning_rate=0.0005)
