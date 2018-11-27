import numpy as np
from scipy.special import expit
from neuron import Neuron

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def softmax_prime(x):

def sigmoid_prime(x):
    return x * (1.0 - x)

class Network:

    def __init__(self, num_inputs, num_hlayers, num_hidden_per_layer, num_outputs,
                 learning_rate, output_activation="softmax", num_iters=200,
                 threshold=0.025, echange_threshold=0.025, relu_multiplier=0.01):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hlayers
        self.num_hidden_per_layer = num_hidden_per_layer
        self.learning_rate = learning_rate
        self.num_iterations = num_iters
        self.threshold = threshold
        self.error_delta_threshold = echange_threshold
        self.output_activation = output_activation
        self.relu_multiplier = relu_multiplier
        self.hidden_layers = []
        num_node_in = self.num_inputs
        for i in range(self.num_hidden_layers):
            self.hidden_layers.append([])
            for j in range(self.num_hidden_per_layer):
                self.hidden_layers[i].append(Neuron(num_node_in, hidden=True, relu_multiplier=relu_multiplier))
            if i == 0:
                num_node_in = self.num_hidden_per_layer
        self.output_layer = [Neuron(num_hidden_per_layer, hidden=False) for i in range(self.num_outputs)]

    """def _evaluate_hidden(self, data, layer, index):
        reduced = np.dot(data, self.hidden_layers[layer][index]["weights"]) + \
            self.hidden_layers[layer][index]["bias"]
        return reduced if reduced > 0 else self.relu_multiplier * reduced"""

    def _evaluate_output(self, input_data):
        pre_active = []
        for i in range(self.num_outputs):
            pre_active.append(self.output_layer[i].evaluate(input_data))#np.dot(input_data, self.output_layer[i]["weights"]) + \
                #self.output_layer[i]["bias"])
        if self.output_activation == "linear":
            return pre_active
        elif self.output_activation == "sigmoid":
            return expit(pre_active)
        elif self.output_activation == "softmax":
            return softmax(pre_active)
        else:
            raise ValueError("Invalid activation key. Should be linear, sigmoid, or softmax.")

    # This function forward propagates on a single data instance
    def forward_propagate(self, data):
        input_data = data
        next_data = []
        for i in range(self.num_hidden_layers):
            for j in range(self.num_hidden_per_layer):
                next_data.append(self.hidden_layers[i][j].evaluate(input_data))
            input_data = next_data
            next_data = []
        return self._evaluate_output(input_data)

    # Algo:
    #  * expected - output
    #  * Multiply by derivative of activation for last layer
    #  * For each previous node:
    #  *   Multiply the errors by the weights corresponding to the current node
    #  *   Multiply the last step's result by the derivative of the hidden layer activation
    #  * For all nodes: final change in error is the dot product of the node's output and its error
    # Link: https://stackoverflow.com/questions/50105249/implementing-back-propagation-using-numpy-and-python-for-cleveland-dataset
    def backpropagate_errors(self, output, expected):
        output_error = expected - output
        o_delta = output_error * self._leaky_relu_derivative(output)
        for i, d in enumerate(o_delta):
            self.output_layer[i].delta = d
        for i in reversed(range(len(self.hidden_layers))):
            for j in range(len(self.hidden_layers[i])):
                if i == len(self.hidden_layers) - 1:
                    for n in range(len(self.output_layer)):
                        self.hidden_layers[i][j].delta = self.output_layer[n].weights[j] * \
                            self.output_layer[n].delta
                else:
                    for n in self.hidden_layers[i+1]:
                        self.hidden_layers[i][j].delta = self.hidden_layers[i+1][n].weights[j] * \
                            self.hidden_layers[i+1][n].delta

    def update_weights(self):
        for i in range(len(self.hidden_layers)):
            for j in range(len(self.hidden_layers[i])):
                print("Hidden Layer {0:d}, Neuron {1:d}".format(i, j))
                self.hidden_layers[i][j].update_weights(self.learning_rate)
        for i in range(len(self.output_layer)):
            print("Output Layer: Neuron {}".format(i))
            self.output_layer[i].update_weights(self.learning_rate)

    def train(self, data, labels):
        # Data assumed to be pre-randomized
        epoch = 0
        prev_error = 0
        while epoch < self.num_iterations:
            error = 0
            for d, l in zip(data, labels):
                expected = np.array([1.0 if i == l else 0.0 for i in range(self.num_outputs)])
                output = self.forward_propagate(d)
                self.backpropagate_errors(output, expected)
                self.update_weights()
                error += sum([(expected[i]-output[i])**2 for i in range(len(expected))])
            print("Epoch {0:d}: Error = {1:f}".format(epoch, error))
            if error < self.threshold:
                print("  Error threshold met. Ending training.")
                break
            if abs(prev_error - error) < self.error_delta_threshold:
                print("  Threshold for change in error met. Ending training.")
                break
            epoch += 1
            prev_error = error
