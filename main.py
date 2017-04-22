
import NeuralNetwork, numpy, collections
import scipy.io, scipy.misc
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.mlab as mlab
import create_mat
import time

WEIGHT_SAVE_FILE = "learned_weights_shift_sgd_reg8.mat"
LOAD_SAVE = True
LEARNING_RATE = 0.1
MOMENTUM = 0.1
NUM_EPOCHS = 5
BATCH_SIZE = 50
REGULARIZATION = 0.2

def save_weights(nn):
	weight_dict = {}
	for i, w in enumerate(nn.weights):
		k = "w%d" % (i + 1)
		weight_dict[k] = w

	scipy.io.savemat(WEIGHT_SAVE_FILE, weight_dict)
	print("Saved weights.")

def plot_cost(iterations, costs):
	plt.plot(iterations, costs)
	plt.tight_layout()
	plt.show(block=False)
	s = input("Press Enter to close...")
	plt.close()

num_inputs = create_mat.COL_SIZE ** 2
output_size = len(create_mat.LETTERS)
hidden_size = 100

nn = NeuralNetwork.NeuralNetwork(input_size=num_inputs, output_size=output_size)

if LOAD_SAVE:
	weights = scipy.io.loadmat(WEIGHT_SAVE_FILE)
	Theta1 = numpy.matrix(weights['w1'])
	Theta2 = numpy.matrix(weights['w2'])

	nn.append_layer(rows=hidden_size, cols=num_inputs + 1,
		            is_output=False, weights=Theta1)
	nn.append_layer(rows=output_size, cols=hidden_size + 1, is_output=True,
					weights=Theta2)
else:
	nn.append_layer(rows=hidden_size, cols=num_inputs + 1,
		            is_output=False)
	nn.append_layer(rows=output_size, cols=hidden_size + 1, is_output=True)

num_examples = 0
y = []

for i, letter in enumerate(create_mat.LETTERS):
	data = scipy.io.loadmat("data/trainingdata_%d.mat" % letter)
	X = data['X']
	X = numpy.matrix(X)
	output_matrix = numpy.matrix(numpy.zeros((output_size, 1)))
	output_matrix[(i, 0)] = 1.
	num_examples += X.shape[0]

	for j, image in enumerate(X):
		y.append(i)
		input_matrix = numpy.matrix(image).reshape((num_inputs, 1))
		nn.add_training_row(input_matrix=input_matrix,
			                output_matrix=output_matrix)

iterations = []
costs = []
cost_raised = 0
cost_slow = 0
factor = 0.1
cost_increased = 0
cost_decreased = 0

print("Initial learning rate: %f" % LEARNING_RATE)

for i in range(NUM_EPOCHS):
	# if i > 0 and i % 100 == 0:
	# 	LEARNING_RATE *= 0.75
	# 	print("Learning rate changed to: %f" % LEARNING_RATE)

	if i > 0 and i % 10 == 0:
		save_weights(nn)
	print("Iteration", (i + 1), "Time:", time.time())
	cost = nn.backward_propagate(learning_rate=LEARNING_RATE,
								 batch_size=BATCH_SIZE, momentum=MOMENTUM,
								 reg_lambda=REGULARIZATION)
	iterations.append(i + 1)
	costs.append(cost)

	if len(costs) >= 2:
		if cost > costs[-2]:
			cost_raised += 1
			cost_slow = 0
		if cost_raised > 2:
			LEARNING_RATE *= (1 - factor)
			cost_decreased += 1
			cost_increased = 0
			cost_raised = 0
			print("Lowered learning rate to %f." % LEARNING_RATE)

		# if cost_decreased % 2 == 0:
		# 	factor *= 0.75
		# 	print("Factor decreased.")

	if len(costs) >= 2:
		if costs[-2] - cost <= 1e-3:
			cost_slow += 1
			# cost_raised = 0
		if cost_slow > 2:
			LEARNING_RATE *= (1 + factor)
			cost_increased += 1
			cost_decreased = 0
			cost_slow = 0
			print("Raised learning rate to %f." % LEARNING_RATE)

		# if cost_increased % 5 == 0:
		# 	factor *= 1.75
		# 	print("Factor increased.")

with open("cost_per_iteration2.txt", "w") as f:
	for i in range(len(iterations)):
		f.write("%d %f\n" % (iterations[i], costs[i]))

save_weights(nn)
plot_cost(iterations, costs)
print("Final cost: %f" % costs[-1])
