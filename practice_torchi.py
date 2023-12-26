# zeros, ones, rand, empty, randint, linspace, eye, 
# multinomial, cat, arange, unsqueeze(learn), masked_fill 
# stack, triu, tril, transpose, softmax, 
# Embedding, Linear functions, data loading
# view, broadcasting semantics, transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn import datasets
import numpy as np
from typing import Any
# think of various matrix operations
# a) Add b) Subtract c) Multiply d) Divide

a = torch.zeros((3, 1))

# print(a)

b = torch.ones((3, 3))

# print(b)

# c = a @ b  #  mat1 and mat2 shapes cannot be multiplied (3x1 and 3x3)
c = a * b
# print(c)

d = torch.rand((4, 3), dtype=torch.float16)

# print(d)

e = torch.triu(input=d)

# print(e)

f = torch.tril(input=d)

# print(f)

g = torch.transpose(input=d, dim0=0, dim1=1)
# print(g)
# print(g.shape)

gt = d.T
# print(gt.shape)

ls = torch.linspace(start=0, end=6, steps=10, dtype=torch.float16)
# print(ls)
vs = ls.view((-1, 2))
# print(vs)
eye = torch.eye(n=5)
# print(eye)

us = vs.unsqueeze(dim=0)
# print(us.shape)
us = vs.unsqueeze(dim=1)
# print(us.shape)
us = vs.unsqueeze(dim=2)
# print(us.shape)

sus = us.squeeze(dim=2)
# print(sus.shape)
sus = sus.squeeze(dim=1)
# print(sus.shape)

susview = sus.view((-1, 10))
# print(susview.shape)

base = torch.ones((5, 5))
# print(base)

make_triu = torch.triu(base)
# print(make_triu)

masked = torch.masked_fill(torch.zeros(5, 5), make_triu == 0, torch.tensor(67))
# print(masked)

# lets get an equation
"""
5x + 2y + 87c = 6
8x + 32y - 25c = 82
756x + 15y + 32c = 75
"""

# make a stack
t1 = torch.tensor([5, 2, 87], dtype=torch.float32)
r1 = torch.tensor([8, 32, -27], dtype=torch.float32)
t2 = torch.tensor([756, 15, 327], dtype=torch.float32)

X = torch.stack((t1, r1, t2))

Y = torch.tensor([6, 82, 76], dtype=torch.float32)
Y = Y.view((3, 1))
# print(Y)

# print(X)

# print(X * Y)

sm = F.softmax(X, dim=1)
# print(sm)

x_data, y_data = datasets.make_regression(n_samples=3, n_features=3, 
                                          noise= 25, random_state=123)
x_torch = torch.from_numpy(x_data.astype(np.float32))
y_torch = torch.from_numpy(y_data.astype(np.float32))
y_torch = y_torch.view((x_torch.shape[0], 1))
print(x_torch.shape)
print(y_torch.shape)


class LinReg(nn.Module):
    def __init__(self, inval, outval):
        super().__init__()
        self.lin = nn.Linear(inval, outval)

    def forward(self, data):
        return self.lin(data)


model = LinReg(inval=3, outval=1)

epoch = 100
learning = 0.01

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning)

for step in range(epoch):
    # forward
    y_pred = model(x_torch)
    # backward
    loss = criterion(y_pred, y_torch)
    # update weights
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # if step % 5 == 0:
        # print(f"step: {step} and loss: {loss.item():.4f}")

# print(model.parameters())

# builiding the dataloader for bigger data set


class RegressionSet(Dataset):
    def __init__(self, regression_data):
        self.x = torch.from_numpy(regression_data[0].astype(np.float32))
        self.y = torch.from_numpy(regression_data[1].astype(np.float32))
        self.samples = self.x.shape[0]

    def __getitem__(self, index) -> Any:
        return self.x[index], self.y[index]

    def __len__(self):
        return self.samples


regr_data = datasets.make_regression(n_samples=50,
                                     n_features=3,
                                     noise=25,
                                     random_state=157)

reg_ds = RegressionSet(regr_data)


# print(reg_ds[0])


reg_dl = DataLoader(reg_ds, 3, True,)

data_iterator = iter(reg_dl)

# print(next(data_iterator))

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning)

for step in range(epoch):
    # loop over the batches and send through model
    for ind, batch in enumerate(reg_dl):
        print(f"send in batch: {ind}")
        # print(batch[0].shape)
        # print(batch[1].shape)
        # forward
        y_batch = model(batch[0])
        y = batch[1].view(batch[0].shape[0], 1)
        # backward
        loss = criterion(y_batch, y)

        # update weights
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    print(f"Step is: {step} and loss is: {loss.item():.4f}")