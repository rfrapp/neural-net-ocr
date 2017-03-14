
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

def save_weights(nn):
	weight_dict = {}
	for i, w in enumerate(nn.weights):
		k = "w%d" % (i + 1)
		weight_dict[k] = w

	scipy.io.savemat("learned_weights5.mat", weight_dict)
	print("Saved weights.")

def plot_cost(iterations, costs):
	plt.plot(iterations, costs)
	plt.tight_layout()
	plt.show(block=False)
	s = input("Press Enter to close...")
	plt.close()

weights = scipy.io.loadmat("learned_weights5.mat")
Theta1 = numpy.matrix(weights['w1'])
Theta2 = numpy.matrix(weights['w2'])

num_inputs = create_mat.COL_SIZE ** 2
# hidden_size = int(len(create_mat.LETTERS) * 2.5)
hidden_size = 36
output_size = len(create_mat.LETTERS)

nn = NeuralNetwork.NeuralNetwork(input_size=num_inputs, output_size=output_size)
# nn.append_layer(rows=hidden_size, cols=num_inputs + 1,
# 	            is_output=False)
# nn.append_layer(rows=output_size, cols=hidden_size + 1, is_output=True)
nn.append_layer(rows=hidden_size, cols=num_inputs + 1,
	            is_output=False, weights=Theta1)
nn.append_layer(rows=output_size, cols=hidden_size + 1, is_output=True,
				weights=Theta2)

num_examples = 0
y = []

for i, letter in enumerate(create_mat.LETTERS):
	data = scipy.io.loadmat("data/trainingdata_%d.mat" % letter)
	X = data['X']
	X = numpy.matrix(X)
	output_matrix = numpy.matrix(numpy.zeros((output_size, 1)))
	output_matrix[(i, 0)] = 1.

	for j, image in enumerate(X):
		y.append(i)
		input_matrix = numpy.matrix(image).reshape((num_inputs, 1))
		nn.add_training_row(input_matrix=input_matrix,
			                output_matrix=output_matrix)
		num_examples += 1

# NeuralNetwork.NeuralNetwork_gradient_check(nn, 1e-4)

iterations = []
costs = []

for i in range(100):
	if i > 0 and i % 10 == 0:
		save_weights(nn)
	print("Iteration", (i + 1))
	cost = nn.backward_propagate(learning_rate=0.007)
	iterations.append(i + 1)
	costs.append(cost)

with open("cost_per_iteration2.txt", "w") as f:
	for i in range(len(iterations)):
		f.write("%d %f\n" % (iterations[i], costs[i]))

save_weights(nn)
plot_cost(iterations, costs)

num_correct = 0
num_incorrect = 0
report = {}
display_img = True
testing_target = -1

for i in range(num_examples):
	result = nn.predict(i)
	target = y[i]

	# print("Result: %d, target: %d" % (result, target))

	if report.get(target) is None:
		report[target] = {"correct": 0, "incorrect": 0}


	if result == target:

		# if target != testing_target:
		# 	if display_img:
		# 		testing_target = target
		# 		s = display_image(nn.input_matrices[i].reshape(create_mat.COL_SIZE, create_mat.COL_SIZE))
		# 		if s == "q":
		# 			break
		# 		# elif s == "c":
		# 		# 	display_img = False

		num_correct += 1
		report[target]["correct"] += 1
	else:
		num_incorrect += 1
		report[target]["incorrect"] += 1

print("Correct: %d, incorrect: %d" % (num_correct, num_incorrect))
pprint(report)
print("Accuracy: %.2f%%" % (num_correct / num_examples * 100))

