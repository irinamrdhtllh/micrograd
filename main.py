import math


class Value:
    def __init__(self, data, _children=(), _operation="", label=""):
        self.data = data
        self.grad = 0.0
        self._prev = _children
        self._op = _operation
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        output = Value(self.data + other.data, (self, other), "+")
        return output

    def __mul__(self, other):
        output = Value(self.data * other.data, (self, other), "*")
        return output

    def tanh(self):
        x = self.data
        tanh = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        output = Value(tanh, (self,), "tanh")
        return output


if __name__ == "__main__":
    # The neuron's inputs
    x1 = Value(2.0)
    x2 = Value(0.0)

    # The neuron's weights
    w1 = Value(-3.0)
    w2 = Value(1.0)

    # The neuron's bias
    b = Value(6.7)

    x1w1 = x1 * w1
    x2w2 = x2 * w2
    xw = x1w1 + x2w2
    n = xw + b
    o = n.tanh()

    print(o)
