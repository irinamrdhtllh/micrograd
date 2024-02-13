from micrograd.engine import Value


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
exp = (2 * n).exp()
o = (exp - 1) / (exp + 1)

# Backward pass
o.backward()

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
