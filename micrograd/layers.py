import random

from micrograd.engine import Value


class Neuron:
    def __init__(self, n_input: int):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_input)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        output = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        output = output.tanh()
        return output

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, n_input: int, n_output: int):
        self.neurons = [Neuron(n_input) for _ in range(n_output)]

    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


class MLP:
    def __init__(self, n_input, n_outputs: list):
        sizes = [n_input] + n_outputs
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(n_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
