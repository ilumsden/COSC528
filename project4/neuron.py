import numpy as np
from scipy.special import expit

class Neuron:

    def __init__(self, num_inputs, hidden=True, relu_multiplier=0.01):
        self.num_inputs = num_inputs
        self.hidden = hidden
        self.weights = np.random.uniform(-0.01, 0.01, size=num_inputs)
        self.bias = np.random.uniform(-0.01, 0.01)
        self.relu_multiplier = relu_multiplier
        self.data = []
        self.delta = 0

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def evaluate(self, input_data):
        self.data = input_data
        reduced = np.dot(input_data, self.weights) + self.bias
        if self.hidden:
            reduced = reduced if reduced > 0 else self.relu_multiplier * reduced
        self.output = reduced
        return reduced

    def reset_training_data(self):
        self.data = []
        self.delta = 0

    def update_weights(self, learning_rate):
        for i, _ in enumerate(self.weights):
            #print("  Old weight:", self.weights[i])
            #print("    Delta = {0:f}, LR = {1:f}, Input = {2:f}".format(self.delta, learning_rate, self.data[i]))
            self.weights[i] += learning_rate * self.delta * self.data[i]
            #print("  New weight:", self.weights[i])
            #print()
            #if not self.hidden:
                #print("  New weight:", self.weights[i])
        self.bias += learning_rate * self.delta
        #if not self.hidden:
            #print("  New bias:", self.bias)
