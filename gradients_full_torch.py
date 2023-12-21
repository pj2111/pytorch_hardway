# 1) Design the model (input, output size and forward pass)
# 2) construct loss and optimizer
# 3) training loop
#  - forward pass: compute prediction
#  - backward pass: compute gradient 
#  - update weights

import torch
import torch.nn as nn
# f = w * x

# f = 2 * x
# the shape of the input is different, when used with torch models
X = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8], [10]], dtype=torch.float32)
# n_features is the cols persent in each datapoint, both inputs & targets
n_samples, n_features = X.shape
x_test = torch.tensor([5],
                      dtype=torch.float32)
# model = nn.Linear(in_features=n_features, out_features=n_features)
# w = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)

# if custom models with different layers have to created, then a class has to be written 
class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(n_features, n_features)
# model prediction, replaced with below py-torch model
# def forward(x):
#   return x * w


# gradient of the loss (MSE)
# MSE = (1/N) * (w*x - y)**2
# dJ/dw = (1/N) * 2x * (w*x - y)
def gradient(x, y, y_pred):
    return np.dot(2 * x, y_pred - y).mean()


loss = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(),
                        lr=0.01)
print(f"prediction before training: f(5)= {model(x_test).item():.3f}")

# training
learning_rate = 0.01
n_iters = 30

for epoch in range(n_iters):
    # prediction to be extracted
    y_pred = model(X)
    # compute loss
    l = loss(Y, y_pred)
    # get gradient ==> backward()
    l.backward()  # will calculate the grad w.r.t 'w'
    # update weights is done by the optimizert
    # with torch.no_grad():
    #    w -= learning_rate * w.grad
    optim.step()
    # make the accumulated grad to 0
    optim.zero_grad()
    # printing epoch
    if epoch % 1 == 0:
        [w, b] = model.parameters()
        # observe how the weights are extracted to display
        print(f"epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss: {l:.8f}")

# do a final prediction
print(f"prediction after training: f(5)= {model(x_test).item():.3f}")
