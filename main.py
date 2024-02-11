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
