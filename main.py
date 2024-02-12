import math


def topological_sort(value):
    if value not in visited:
        visited.add(value)

        for child in value._prev:
            topological_sort(child)

        stack.append(value)


class Value:
    def __init__(self, data, _children=(), _operation="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = _children
        self._op = _operation
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        def _backward():
            self.grad = 1.0 * output.grad
            other.grad = 1.0 * output.grad

        output = Value(self.data + other.data, (self, other), "+")
        output._backward = _backward

        return output

    def __mul__(self, other):
        def _backward():
            self.grad = other.data * output.grad
            other.grad = self.data * output.grad

        output = Value(self.data * other.data, (self, other), "*")
        output._backward = _backward

        return output

    def tanh(self):
        def _backward():
            self.grad = (1 - tanh**2) * output.grad

        x = self.data
        tanh = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        output = Value(tanh, (self,), "tanh")
        output._backward = _backward

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

    # Forward pass
    x1w1 = x1 * w1
    x2w2 = x2 * w2
    xw = x1w1 + x2w2
    n = xw + b
    o = n.tanh()

    stack = []
    visited = set()
    topological_sort(o)
    print(stack)

    # Set the gradient of the output
    o.grad = 1

    # Backward pass
    for node in reversed(stack):
        node._backward()

    print(f"Gradient of o: {o.grad}")
    print(f"Gradient of n: {n.grad}")
    print(f"Gradient of xw: {xw.grad}")
    print(f"Gradient of b: {b.grad}")
    print(f"Gradient of x1w1: {x1w1.grad}")
    print(f"Gradient of x2w2: {x2w2.grad}")
    print(f"Gradient of x1: {x1.grad}")
    print(f"Gradient of w1: {w1.grad}")
    print(f"Gradient of x2: {x2.grad}")
    print(f"Gradient of w2: {w2.grad}")
