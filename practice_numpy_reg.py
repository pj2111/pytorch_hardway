# get the data
# do train test split
# build the model
# declare the loss criterion
# training loop
    # get prediction from training
    # calculate losses
    # update weights

import torch
import torch.nn as nn
import numpy as np

x_train = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y_train = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
test = torch.tensor([5], dtype=torch.float32)
weight = torch.tensor([0.], dtype=torch.float32, requires_grad=True)
learning_rate = 0.01
epochs = 30
in_vars = x_train.shape[1]
out_vars = 1

class Linreg(nn.Module):
    def __init__(self, in_param, out_param):
        super(Linreg, self).__init__()
        self.linear = nn.Linear(in_param, out_param)

    def forward(self, x):
        return self.linear(x)


model = Linreg(in_vars, out_vars)
criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(f"prediction of the model before training: {model(test).item():.2f}")

for x in range(epochs):
    # forward
    y_pred = model(x_train)
    # loss
    # loss = criterion(y_pred, y_train)
    loss = criterion(y_pred, y_train)
    # update weights
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # output stat
    print(f"Epoch: {x} Loss: {loss.item():.8f}")


print(f"prediction of the model after training: {model(test).item():.3f}")