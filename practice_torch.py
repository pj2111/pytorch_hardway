import torch
from torch import nn
import numpy as np
from sklearn import datasets

data = datasets.make_regression(n_samples=100,
                                n_features=4,
                                n_targets=1,
                                noise=50,
                                random_state=1233)
# print(data)
x_data = data[0]
y_data = data[1]

X = torch.from_numpy(x_data.astype(np.float32))
Y = torch.from_numpy(y_data.astype(np.float32))
Y = Y.view(-1, 1)
# print(Y.size())

lin = nn.Linear(in_features=4, out_features=1)

for name, param in lin.named_parameters():
    print(name, param)
print(lin.weight)
loss = nn.MSELoss()

# Forward prop
y_prop = lin(X)
loss_calc = loss(y_prop, Y)

# back prop on the calculated loss
loss_calc.backward()

# looking at the parameters
for name, param in lin.named_parameters():
    print(name, param)