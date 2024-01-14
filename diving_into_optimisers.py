# Script that dives into the details of Optimisers

import torch
from torch import nn
from torch import optim
import math


a = torch.linspace(start=0, end=2 * math.pi, steps=25, requires_grad=True)
b = torch.sin(a)
c = 2 * b
d = c.sum() 

print("tensor a: ", a)
print("tensor b: ", b)
print("tensor c: ", c)
print("tensor d: ", d)

print("Variety of next functions to calculate the backward prop")

d.backward()

print("Gradients calculated at leaves only: ", a.grad)

print("Grad function at d node: ", d.grad_fn.next_functions[0][0])
print("Grad function at c node: ", d.grad_fn.next_functions[0][0].next_functions[0][0])
print("Grad function at b node: ", d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0])
# a node won't have the gradient


class TinyMod(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(TinyMod, self).__init__(*args, **kwargs)

        self.layer1 = nn.Linear(1000, 100)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        return self.layer2(out)


# model specs
BATCH_SIZE = 16
DIM_IN = 1000
HIDDEN_SIZE = 100
DIM_OUT = 10

# data in and out
some_in = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
some_out = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)

# model declaration
model = TinyMod()

print(model.layer2.weight[0][:10])
# print(model.layer2.weight.grad)

# starting optimiser work

sgd_optimiser = optim.SGD(model.parameters(), lr=0.001)

pred = model(some_in)

loss = (some_out - pred).pow(2).sum()
# print(loss)

# doing the back propagation
loss.backward()
# print(model.layer2.weight[0][:10])
print(model.layer2.weight.grad[0][:10])


for ep in range(5):
    pred = model(some_in)
    loss = (some_out - pred).pow(2).sum()
    loss.backward()

print("Gradient after running loss backward 5 times")
print(model.layer2.weight.grad[0][:10])


print("Weights after running loss backward 5 times. No Change")
print(model.layer2.weight[0][:10])

# getting optimiser to step
sgd_optimiser.step()
# print(model.layer2.weight.grad[0][:10])
print("Weights after running optimiser 1 times.Change")
print(model.layer2.weight[0][:10])
