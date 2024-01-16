# Script that dives into the details of Optimisers
import numpy as np
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

def SGD(data, batch_size, lr):
    N = len(data)
    np.random.shuffle(data)
    mini_batches = np.array([data[i:i+batch_size]
    for i in range(0, N, batch_size)]):
        for X,y in mini_batches:
            backprop(X, y, lr) 

sgd_optimiser = optim.SGD(model.parameters(), lr=0.001)
"""
SGD is not used much these days as it get stuck in local minima.
Adagrad, Adadelta, RMSprop, and ADAM generally handle saddle points better.
SGD with momentum renders some speed to the optimization and also helps escape local minima better.
Averaged Stochastic Gradient Descent(ASGD) algorithm
"""

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


# Starting the code from Ultimate Optimizers

# Adam is One of the most popular optimizers
# also known as adaptive Moment Estimation, it combines 
# the good properties of Adadelta and RMSprop optimizer 
# into one and hence tends to do better for most of the problems
adam_optimiser = optim.Adam(model.parameters(), lr=0.001,
                            betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0,
                            amsgrad=False)

#  practically that AdamW yields better training loss,
# that means the models generalize much better than
# models trained with Adam allowing the remake to
# compete with stochastic gradient descent with momentum.
adamw_optimiser = optim.AdamW(params=model.parameters(),
                              lr=0.001, betas=(0.9, 0.999),
                              eps=1e-8, weight_decay=0.01,
                              amsgrad=False)

# SparseAdam Implements a lazy version of Adam algorithm which is suitable for sparse tensors.
# In this variant of adam optimizer, only moments that show up in the gradient get updated,
# and only those portions of the gradient get applied to the parameters.
sparse_adam = optim.SparseAdam(params=model.parameters(),
                               lr=0.001, betas=(0.9, 0.999),
                               eps=1e-08)

# Ada-Delta Optimiser Code
# method dynamically adapts over time using only first order information and 
# has minimal computational overhead beyond vanilla stochastic gradient 
# descent. The method requires no manual tuning of a learning rate and
# appears robust to noisy gradient information, different model architecture 
# choices, various data modalities and selection of hyperparameters.
def Adadelta(weights, sqrs, deltas, rho, batch_size):
    eps_stable = 1e-5
    for weight, sqr, delta in zip(weights, sqrs, deltas):
        g = weight.grad / batch_size
        sqr[:] = rho * sqr + (1. - rho) * np.square(g)
        cur_delta = np.sqrt(delta + eps_stable) / np.sqrt(sqr + eps_stable) * g
        delta[:] = rho * delta + (1. - rho) * cur_delta * cur_delta
        # update weight in place.
        weight[:] -= cur_delta

"""
rho (float, optional) – coefficient used for computing a running average
of squared gradients (default: 0.9)

eps (float, optional) – term added to the denominator to improve
numerical stability (default: 1e-6)

lr (float, optional) – coefficient that scale delta before it
is applied to the parameters (default: 1.0)

weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)
"""
adadel_optimiser = optim.Adadelta(params=model.parameters(),
                                  lr=1.0, rho=0.9, eps=1e-6,
                                  weight_decay=0)

"""
penalizes the learning rate for parameters that are frequently updated, 
instead, it gives more learning rate to sparse parameters, 
parameters that are not updated as frequently.
"""
def AdaGrad(data, theta, num_iters, weights, eps, lr):
    grad_sums = np.zeros(theta.shape[0])
    for t in range(num_iters):
        grads = compute_gradients(data, weights)
        grad_sums += grads ** 2
        grad_updt = grads / (np.sqrt(grad_sums + eps))
        weights = weights - lr * grad_updt
    return weights


adagrad_optim = optim.Adagrad(params=model.parameters(), lr=0.01,
                              lr_decay=0, weight_decay=0,
                              initial_accumulator_value=0,
                              eps=1e-10)


lbgfs_optimiser = optim.LBFGS(model.parameters(),
                              lr=1, max_iter=20, max_eval=None,
                              tolerance_grad=1e-07, tolerance_change=1e-09,
                              history_size=100, line_search_fn=None)

rmsprop = optim.RMSprop(model.parameters(),
                        lr=0.01, alpha=0.99, eps=1e-08,
                        weight_decay=0,
                        momentum=0, centered=False)
