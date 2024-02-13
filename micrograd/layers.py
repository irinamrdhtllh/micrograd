import random

from micrograd.engine import Value


class Neuron:
    def __init__(self, n_input: int):
        self.weights = (Value(random.uniform(-1, 1)) for _ in range(n_input))
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        output = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        output = output.tanh()
        return output


class Layer:
    def __init__(self, n_input: int, n_output: int):
        self.neurons = [Neuron(n_input) for _ in range(n_output)]

    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs


class MLP:
    def __init__(self, n_input, n_outputs: list):
        sizes = [n_input] + n_outputs
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(n_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    x = [2.0, 3.0, -1]
    nn = MLP(n_input=2, n_outputs=[4, 4, 1])
    print(nn(x))
