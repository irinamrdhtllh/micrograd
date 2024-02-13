from micrograd.engine import Value
from micrograd.layers import MLP


inputs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
targets = [1.0, -1.0, -1.0, 1.0]
nn = MLP(3, [4, 4, 1])

pred_targets = [nn(x) for x in inputs]

loss = sum(
    ((y_pred - y_true) ** 2 for y_true, y_pred in zip(targets, pred_targets)),
    start=Value(0),
)

loss.backward()

print(nn.layers[0].neurons[0].weights[0].grad)
