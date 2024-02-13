import math


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
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad

        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), "+")
        output._backward = _backward

        return output

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), "*")
        output._backward = _backward

        return output

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        def _backward():
            self.grad = other * (self.data ** (other - 1)) * output.grad

        assert isinstance(other, (int, float)), "Only support int or float powers"
        output = Value(self.data**other, (self,), f"**{other}")
        output._backward = _backward

        return output

    def __truediv__(self, other):
        return self * other**-1

    def exp(self):
        def _backward():
            self.grad += output.data * output.grad

        x = self.data
        output = Value(math.exp(x), (self,), "exp")
        output._backward = _backward

        return output

    def tanh(self):
        def _backward():
            self.grad += (1 - tanh**2) * output.grad

        x = self.data
        tanh = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        output = Value(tanh, (self,), "tanh")
        output._backward = _backward

        return output

    def backward(self):
        def topological_sort(value):
            if value not in visited:
                visited.add(value)

                for child in value._prev:
                    topological_sort(child)

                stack.append(value)

        stack = []
        visited = set()
        topological_sort(self)

        # Backward pass
        self.grad = 1.0
        for node in reversed(stack):
            node._backward()
