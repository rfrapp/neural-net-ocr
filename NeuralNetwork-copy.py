
import numpy, math, copy, random, sys
from scipy.special import expit
import scipy.io
import scipy.optimize
import sklearn.utils


def softmax(w):
    e = numpy.exp(w - numpy.amax(w))
    dist = e / numpy.sum(e)
    return dist


class NeuralNetwork(object):
    def __init__(self, input_size, output_size,
                 activation_function=None, dactivation_function=None,
                 cost_function=None, dcost_function=None):
        self.weights = []
        self.input_size = input_size
        self.output_size = output_size
        self.input_matrices = []
        self.output_matrices = []
        self.activations = []
        self.sizes = []
        self.initial_weights = []

        # Add the first layer to the network.
        empty_activation = numpy.matrix([1] + ([0] * input_size))
        empty_activation = numpy.reshape(empty_activation, (input_size + 1, 1))
        self.activations.append(empty_activation)

        if not activation_function:
            self.activation_function = lambda z: expit(z)
            # self.activation_function =
            # self.activation_function = lambda z: 1 / (1 + numpy.exp(-z))
            # self.activation_function = lambda z: math.exp(-numpy.logaddexp(0, -z))
        else:
            self.activation_function = activation_function
        self.activation_function = numpy.vectorize(self.activation_function)

        if not dactivation_function:
            self.dactivation_function = lambda a: a * (1 - a)
        else:
            self.dactivation_function = dactivation_function
        self.dactivation_function = numpy.vectorize(self.dactivation_function)

        # t = Target value
        # y = input value
        if not cost_function:
            self.cost_function = lambda t, y: (1 / 2) * (t - y) ** 2
        else:
            self.cost_function = cost_function

        # t = Target value
        # y = input value
        if not dcost_function:
            self.dcost_function = lambda t, y: y - t
        else:
            self.dcost_function = dcost_function
        self.dcost_function = numpy.vectorize(self.dcost_function)

    def add_training_row(self, input_matrix, output_matrix):
        '''Adds a training example to the network.
           Parameters:
               - input_matrix = A matrix of size self.input_size x 1
               - output_matrix = A matrix of size self.output_size x 1'''
        if not isinstance(input_matrix, numpy.matrix):
            raise ValueError("The input must be of type numpy.matrix")
        if not isinstance(output_matrix, numpy.matrix):
            raise ValueError("The output must be of type numpy.matrix")

        i_m, _ = input_matrix.shape
        o_m, _ = output_matrix.shape

        if i_m != self.input_size:
            raise ValueError("The input matrix must be of size %d x 1" % self.input_size)
        if o_m != self.output_size:
            raise ValueError("The output matrix must be of size %d x 1" % self.output_size)

        self.input_matrices.append(input_matrix)
        self.output_matrices.append(output_matrix)

    def set_input(self, j):
        '''This creates an m x 1 matrix in the first activation and sets its
           values to the ones in the training example at index j.
           Paramters:
               - j = The index of the training example to use'''
        a = numpy.insert(self.input_matrices[j], 0, 1, axis=0)
        self.activations[0] = a

    def append_layer(self, rows, cols, is_output, weights=None):
        '''Adds layer of weights and activations to the network. The new
               activation matrix that's added includes a 1 as its first element
               (for a future layer's bias) while setting the rest to 0.
           Parameters:
               - rows = The number of rows in this layer's weight matrix. This
                        should correspond to the columns in the previous
                        layer's wieght matrix
               - cols = The number of outputs in this layer
               - cols = The number of columns in
               - weights = Optional param to specify the wieghts of this
                           layer. If not specified, then random values between
                           0 and 1 are used'''
        if weights is not None and not isinstance(weights, numpy.matrix):
            raise ValueError("Input weights must be of type numpy.matrix")

        self.sizes.append((rows, cols))

        prev_m, _ = self.activations[-1].shape

        if prev_m == cols:
            if weights is None:
                rand_values = numpy.random.uniform(-0.2, 0.2, (rows, cols))
                weights = numpy.reshape(rand_values, (rows, cols))
            self.weights.append(weights)
            self.initial_weights.append(weights)
            if not is_output:
                empty_activation = numpy.matrix([1.] + ([0.] * (rows)))
                empty_activation = empty_activation.reshape(rows + 1, 1)
            else:
                empty_activation = numpy.matrix(([0.] * rows))
                empty_activation = empty_activation.reshape(rows, 1)

            self.activations.append(empty_activation)
        else:
            raise ValueError("Input weights must have %d rows" % (prev_m))

    def shuffle_examples(self):
        self.input_matrices, self.output_matrices = sklearn.utils.shuffle(self.input_matrices, self.output_matrices)

    def forward_propagate(self):
        """Propagates the input throught the network. A prerequisite for this
           function call is that you've already called set_input() to set
           the desired input to propagate through the network."""
        for i, weights in enumerate(self.weights):
            z = weights * self.activations[i]
            self.z = z
            if i != len(self.weights) - 1:
                a = self.activation_function(z)
            else:
                a = softmax(z)
            self.a = a
            self.w = weights
            m, _ = a.shape

            if i + 1 < len(self.weights):
                a = numpy.insert(a, 0, 1, axis=0)
                self.activations[i + 1] = a
            else:
                a = a.reshape((m, 1))
                self.activations[i + 1] = a

    def backward_propagate(self, learning_rate=0.5, batch_size=None):
        '''Performs backpropagation with all of the current training data. The
               weights are updated using Gradient Descent with the input
               learning rate.
           Parameters:
               - learning_rate = The learning rate to be used in Gradient
                                 Descent when updating the weights (i.e.
                                 weight = weight - learning_rate * gradient
           Returns:
                The normalized cost before backprop was run on this network.'''
        if batch_size is None:
            batch_size = len(self.input_matrices)
        else:
            self.shuffle_examples()

        num_layers = len(self.activations)
        a_m, _ = self.activations[num_layers - 1].shape
        Delta = [None for a in self.activations]
        delta = [None for a in self.activations]

        costJ = 0.0

        for k in range(0, len(self.input_matrices), batch_size):
            for j in range(k, k + batch_size):
                self.set_input(j)
                self.forward_propagate()

                # Compute the cost for this row of data.
                j_sum = 0.0
                for q in range(self.output_size):
                    y_out = self.output_matrices[j][(q, 0)]
                    a_out = self.activations[-1][(q, 0)]
                    c = 0.0
                    # if a_out == 0. or a_out == 1.:
                    #     print("Something's wrong. Saving file")
                    #     print(self.activations[-1])
                    #     raise ValueError("You done goofed.")
                    try:
                        c = -y_out * math.log(a_out) - (1 - y_out) * math.log(1 - a_out)
                        j_sum += c
                    except:
                        print(self.activations[-1])
                        raise ValueError("You done goofed")
                costJ += j_sum

                # Calculate the error with respect to the output of each node in
                # the last layer.
                a_values = self.activations[-1]
                dA = a_values - self.output_matrices[j]
                delta[-1] = dA

                if Delta[-1] is None:
                    Delta[-1] = delta[-1] * self.activations[-2].T
                else:
                    Delta[-1] += delta[-1] * self.activations[-2].T

                # Calculate delta values for each hidden layer (this of course
                # excludes the input layer).
                for l in range(num_layers - 2, 0, -1):
                    # Calculate the change in the activation function for each
                    # node in this layer with respect to the input for the
                    # activation function.
                    activ_m = self.dactivation_function(self.activations[l])
                    activ_m = activ_m[1:]

                    # Calculate the total change in the cost function with respect
                    # to each of the nodes from the current layer (this is the
                    # "self.weights[l] * delta[l + 1]" portion. This expression
                    # sums up the total errors for each node that a node in
                    # the current layer connects to), then multiply it by the above
                    # term to get the delta for this layer.
                    weights = self.weights[l].T[1:]
                    delta[l] = numpy.multiply(weights * delta[l + 1], activ_m)

                    # Calculate the gradient for each weight in this layer by
                    # multiplying the activation value of the node which this weight
                    # connects to in the previous layer with the error in the
                    # current layer.
                    if Delta[l] is None:
                        Delta[l] = delta[l] * self.activations[l - 1].T
                    else:
                        Delta[l] += delta[l] * self.activations[l - 1].T

                # Normalize the error.
                for i in range(1, num_layers - 1):
                    Delta[i] = (1 / len(self.input_matrices)) * Delta[i]

                # Update the weights.
                for i in range(len(self.weights)):
                    self.weights[i] -= learning_rate * Delta[i + 1]

        # Return a normalized cost.
        return costJ / len(self.input_matrices)

    def predict(self, k):
        '''Predict the class of the kth training set.
           Parameters:
               - k = The index of a row in the training set. This function will
                     predict the class of this row
           Returns:
               An integer which represents the class of the input row. This
               will be a number from 0 to output_size (exclusive)'''
        self.set_input(k)
        self.forward_propagate()

        output = self.activations[-1].ravel().tolist()[0]
        return output.index(max(output))

    def predict_new(self, input_matrix):
        a = numpy.insert(input_matrix, 0, 1, axis=0)
        self.activations[0] = a

        self.forward_propagate()

        output = self.activations[-1].ravel().tolist()[0]
        return output.index(max(output))

def NeuralNetwork_cost(nn):
    '''Returns the total cost of the training data.'''
    J = 0.0

    for i in range(len(nn.input_matrices)):
        k_sum = 0.0
        nn.set_input(i)
        nn.forward_propagate()

        for k in range(nn.output_size):
            y_out = nn.output_matrices[i][(k, 0)]
            a_out = nn.activations[-1][(k, 0)]
            k_sum += -y_out * math.log(a_out) - (1 - y_out) * math.log(1 - a_out)

        J += k_sum

    return J / len(nn.input_matrices)

def reconstruct_weights(theta, sizes):
    if isinstance(theta, numpy.matrix):
        theta = theta.ravel().tolist()[0]

    weights = []
    p = 0
    for row, col in sizes:
        w = numpy.matrix(theta[p:p + row * col]).reshape((row, col))
        weights.append(w)
        p += row * col
    return weights

def collapse_weights(nn):
    theta = []
    for weights in nn.weights:
        theta += weights.ravel().tolist()[0]
    return theta

def NeuralNetwork_gradient_check(nn, epsilon):
    theta_list = collapse_weights(nn)
    theta = numpy.matrix(theta_list).reshape((len(theta_list), 1))
    numgrad = numpy.zeros((len(theta_list), 1))

    for i, w in enumerate(theta):
        print("%d out of %d" % (i, len(theta_list)))

        theta[(i, 0)] -= epsilon
        weights = reconstruct_weights(theta, nn.sizes)
        nn.weights = weights
        loss0 = NeuralNetwork_cost(nn)

        theta[(i, 0)] += 2 * epsilon
        weights = reconstruct_weights(theta, nn.sizes)
        nn.weights = weights
        loss1 = NeuralNetwork_cost(nn)

        theta[(i, 0)] = 0

        numgrad[(i, 0)] = (loss1 - loss0) / (2 * epsilon)

    print(numgrad)
