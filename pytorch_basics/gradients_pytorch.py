# Weights are updated so gradients can be stored, 
# for the data, gradients are not required as they are constants
# model, loss functions are still the same
import torch
# f = w * x

# f = 2 * x

X = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8, 10], dtype=torch.float32)
w = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)

# model prediction
def forward(x):
    return x * w

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

# gradient of the loss (MSE)
# MSE = (1/N) * (w*x - y)**2
# dJ/dw = (1/N) * 2x * (w*x - y)
def gradient(x, y, y_pred):
    return np.dot(2 * x, y_pred - y).mean()


print(f"prediction before training: f(5)= {forward(5):.3f}")

# training
learning_rate = 0.01
n_iters = 30

for epoch in range(n_iters):
    # prediction to be extracted
    y_pred = forward(X)
    # compute loss
    l = loss(Y, y_pred)
    # get gradient ==> backward()
    l.backward()  # will calculate the grad w.r.t 'w'
    # weights are used for calculating the y_hat, which is
    # part of the loss calculation. (review the back_propagation.drawio)
    # update weight, with the gradients calculated by backward() method
    with torch.no_grad():
        w -= learning_rate * w.grad
    # make the accumulated grad to 0
    w.grad.zero_()
    # printing epoch
    if epoch % 1 == 0:
        print(f"epoch {epoch + 1}: w = {w:.3f}, loss: {l:.8f}")

# do a final prediction
print(f"prediction after training: f(5)= {forward(5):.3f}")