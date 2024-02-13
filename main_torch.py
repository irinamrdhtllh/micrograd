import torch


# The neuron's inputs
x1 = torch.Tensor([2.0]).double()
x1.requires_grad = True
x2 = torch.Tensor([0.0]).double()
x2.requires_grad = True

# The neuron's weights
w1 = torch.Tensor([-3.0]).double()
w1.requires_grad = True
w2 = torch.Tensor([1.0]).double()
w2.requires_grad = True

# The neuron's bias
b = torch.Tensor([6.7]).double()
b.requires_grad = True

# Forward pass
n = x1 * w1 + x2 * w2 + b
o = torch.tanh(n)

# Backward pass
o.backward()

print(f"Gradient of o: {o.grad}")
print(f"Gradient of n: {n.grad}")
print(f"Gradient of b: {b.grad}")
print(f"Gradient of x1: {x1.grad}")
print(f"Gradient of w1: {w1.grad}")
print(f"Gradient of x2: {x2.grad}")
print(f"Gradient of w2: {w2.grad}")
