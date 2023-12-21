# 1) Designing the model
# 2) Assigning the Loss Function & optimizer
# 3) Training Loop
#   - Forward Pass : Predicting value
#   - Backward Pass: Predicting gradient & optimizing params
#   - Updating weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
# prep the datasets
X_numpy, y_numpy = datasets.make_regression(n_samples=100,
                                            n_features=1,
                                            noise=20,
                                            random_state=1)
# look at the data, and understand its shape and size
# print(X_numpy[:3])  # [[-0.61175641] [-0.24937038] [ 0.48851815]]
# print(y_numpy[:3])  # [-55.5385928  -10.66198475  22.7574081 ]
# convert the numpy to torch
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
lr = 0.01
print(X.shape)  # torch.Size([100, 1])
print(y.shape)  # torch.Size([100])

# the shape of the y data needs to be same as x data
y = y.view(y.shape[0], 1)
print(y.shape)  # torch.Size([100, 1])

n_samples, n_features = X.shape  # 100, 1
input_size, output_size = n_features, n_features
# 1
model = nn.Linear(input_size, output_size)

# 2
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr)

# Training loop
epochs = 100
for epo in range(epochs):
    # forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)
    # backward pass
    loss.backward()
    # update the parameter
    optimizer.step()
    # empty the gradient
    optimizer.zero_grad()

    if epo + 1 % 10 == 0:
        print(f"epoch: {epo}, loss = {loss.item():.4f}")

# create a plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
