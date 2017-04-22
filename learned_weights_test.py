
import NeuralNetwork, numpy, collections
import scipy.io, scipy.misc
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.mlab as mlab
import create_mat

weights = scipy.io.loadmat("learned_weights_shift_sgd_reg8.mat")
Theta1 = numpy.matrix(weights['w1'])
Theta2 = numpy.matrix(weights['w2'])

num_inputs = create_mat.COL_SIZE ** 2
hidden_size = 100
output_size = len(create_mat.LETTERS)

nn = NeuralNetwork.NeuralNetwork(input_size=num_inputs, output_size=output_size)
nn.append_layer(rows=hidden_size, cols=num_inputs + 1,
	            is_output=False, weights=Theta1)
nn.append_layer(rows=output_size, cols=hidden_size + 1, is_output=True,
				weights=Theta2)

y = []

num_training_examples = 0
num_correct_training = 0
num_incorrect_training = 0

num_testing_examples = 0
num_correct_testing = 0
num_incorrect_testing = 0

training_report = {}
testing_report = {}
display_img = True
testing_target = -1

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
		num_training_examples += 1

for i in range(num_training_examples):
	result = nn.predict(i)
	target = y[i]

	if training_report.get(target) is None:
		training_report[target] = {"correct": 0, "incorrect": 0}

	if result == target:
		num_correct_training += 1
		training_report[target]["correct"] += 1
	else:
		num_incorrect_training += 1
		training_report[target]["incorrect"] += 1

print("Training:")
print("Correct: %d, incorrect: %d" % (num_correct_training, num_incorrect_training))
pprint(training_report)
print("Accuracy: %.2f%%" % (num_correct_training / num_training_examples * 100))

for i, letter in enumerate(create_mat.LETTERS):
	data = scipy.io.loadmat("data/testingdata_%d.mat" % letter)
	X = data['X']
	X = numpy.matrix(X)

	if testing_report.get(i) is None:
		testing_report[i] = {"correct": 0, "incorrect": 0}

	for j, image in enumerate(X):
		input_matrix = numpy.matrix(image).reshape((num_inputs, 1))
		predicted = nn.predict_new(input_matrix)
		num_testing_examples += 1

		if predicted == i:
			num_correct_testing += 1
			testing_report[i]["correct"] += 1
		else:
			num_incorrect_testing += 1
			testing_report[i]["incorrect"] += 1

print("Testing:")
print("Correct: %d, incorrect: %d" % (num_correct_testing, num_incorrect_training))
pprint(testing_report)
print("Accuracy: %.2f%%" % (num_correct_testing / num_testing_examples * 100))

# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------

