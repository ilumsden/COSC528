import numpy as np
from scipy.special import expit, softmax

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
                hidden_neuron = {"weights": np.random.uniform(size=num_node_in),
                                 "bias": np.random.uniform()}
                self.hidden_layers[i].append(hidden_neuron)
            if i == 0:
                num_node_in = self.num_hidden_per_layer
        self.output_layer = [{"weights": np.random.uniform(size=self.num_hidden_per_layer),
                              "bias": np.random.uniform()} for i in range(self.num_outputs)]

    def _evaluate_hidden(self, data, layer, index):
        reduced = np.dot(data, self.hidden_layers[layer][index]["weights"]) + \
            self.hidden_layers[layer][index]["bias"]
        return reduced if reduced > 0 else self.relu_multiplier * reduced

    def _evaluate_output(self, input_data):
        pre_active = []
        for i in range(self.num_outputs):
            pre_active.append(np.dot(input_data, self.output_layer[i]["weights"]) + \
                self.output_layer[i]["bias"])
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
                next_data.append(self._evaluate_hidden(input_data, i, j))
            input_data = next_data
            next_data = []
        return self._evaluate_output(input_data)
