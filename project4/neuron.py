import numpy as np
from scipy.special import expit

class Neuron:

    def __init__(self, num_inputs, hidden=True, relu_multiplier=0.01):
        self.num_inputs = num_inputs
        self.hidden = hidden
        self.weights = np.random.uniform(size=num_inputs)
        self.bias = np.random.uniform()
        self.relu_multiplier = relu_multiplier
        self.data = []

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def evaluate(self, input_data):
        self.data = input_data
        reduced = np.dot(input_data, self.weights) + self.bias
        if hidden:
            reduced = reduced if reduced > 0 else self.relu_multiplier * reduced
        return reduced

    def reset_input_data(self):
        self.data = []

    def update_weights(self, learning_rate, error):
        for i, _ in enumerate(self.weights):
            self.weights[i] += learning_rate * error * self.data[i]