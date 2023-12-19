import numpy as np

# f = w * x

# f = 2 * x

X = np.array([1, 2, 3, 4, 5], dtype=np.float32)
Y = np.array([2, 4, 6, 8, 10], dtype=np.float32)
w = 0.0

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
n_iters = 10

for epoch in range(n_iters):
    # prediction to be extracted
    y_pred = forward(X)
    # compute loss
    l = loss(Y, y_pred)
    # get gradient
    dw = gradient(X, Y, y_pred)
    # update weight
    w -= learning_rate * dw
    # printing epoch
    if epoch % 1 == 0:
        print(f"epoch {epoch + 1}: w = {w:.3f}, loss: {l:.8f}")

# do a final prediction
print(f"prediction after training: f(5)= {forward(5):.3f}")