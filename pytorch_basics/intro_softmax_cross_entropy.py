import torch
import torch.nn as nn
import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


x = np.array([2.0, 1.0, 0.1])
output = softmax(x)
# print('numpy softmax: ', output)


tensor_x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(tensor_x, dim=0)
# print(outputs)


def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

# y is one-hot encoded
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
Y = np.array([1, 0, 0])

Y_pred_awes = np.array([0.7, 0.2, 0.1])
Y_pred_worse = np.array([0.1, 0.3, 0.6])

l1 = cross_entropy(Y, Y_pred_awes)
l2 = cross_entropy(Y, Y_pred_worse)

# print(f"Worse prediction: {l2}")
# print(f"Awesome prediction: {l1}")

# implementing through numpy
loss = nn.CrossEntropyLoss()

Y_torch = torch.tensor([0])
# n samples X n classes = 1 X 3
Y_pred = torch.tensor([[2.0, 1.0, 0.1]])
Y_bad = torch.tensor([[0.5, 2.0, 0.3]])

# getting RuntimeError: "log_softmax_lastdim_kernel_impl" not implemented for 'Long'
# need to provide the prediction as 1st arg, and actual as 2nd arg
l1 = loss(Y_pred, Y_torch)
# print(l1.item())

# implementing the same for multi-class prediction

Y = torch.tensor([2, 0, 1])  # No softmax layer or the classes are not "one-hot encoded."
y_pred_good = torch.tensor([[0.1, 1.0, 2.5], [2.0, 1.0, 0.1], [0.5, 2.7, 0.2]])

l2 = loss(y_pred_good, Y)
print(l2.item())
