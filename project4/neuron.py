from scipy.special import expit, softmax
import numpy as np

class Neuron:

    def __init__(self, num_inputs, output_node=False, activation="sigmoid", relu_bound=0.01):
        self.num_inputs = num_inputs
        self.output_node = output_node
        self.weights = np.random.rand(num_inputs, 1)
        self.bias = np.random.randint(0, 1)
        if output_node:
            if activation == "softmax":
                self.activation_key = "linear"
            else:
                self.activation_key = activation
        else:
            self.activation_key = "relu"
        self.alpha = abs(relu_bound)

    def activate(self, reduced):
        if self.activation_key == "linear":
            return reduced
        elif self.activation_key == "sigmoid":
            return expit(reduced)
        elif self.activation_key == "relu":
            return reduced if reduced > self.alpha*reduced else self.alpha * reduced
        else:
            raise ValueError("Invalid activation key")

    def calculate(self, x):
        reduced = np.dot(self.weights, x) + self.bias
        return self.activate(reduced)
