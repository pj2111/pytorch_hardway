import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

train_path = "D:\\gitFolders\\pytorch_hardway\\data\\train.csv"
data = pd.read_csv(train_path)

# making data to numpy

data = np.array(data)
# print(data.shape)  # (42000, 785)
m, n = data.shape
# getting the data ready for splitting
np.random.shuffle(data)
print(data[1, :])

data_dev = data[0:1000].T  # 
print(data_dev[0])
Y_dev = data_dev[0]  # All the labels
X_dev = data_dev[1:n]  # All the features
X_dev = X_dev / 255.  # normalize the features

data_train = data[1000:m].T
Y_train = data_train[0]  # All the labels
X_train = data_train[1:n]  # all the features, in the columns
X_train = X_train / 255.  # normalize the features

# output layer will have 10 units corresponding to the ten digit classes with softmax activation.


def init_params():
    # getting the weights and biases intiated
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, W2, b1, b2


def ReLU(Z):
    # if less than 0, then return 0, 
    # else return the value
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def ReLU_deriv(Z):
    return Z > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)  # Why do one hot?
    dZ2 = A2 - one_hot_Y  #
    dW2 = 1 / m * dZ2.dot(A1.T)  #
    db2 = 1 / m * np.sum(dZ2)  #
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)  #
    dW1 = 1 / m * dZ1.dot(X.T)  #
    db1 = 1 / m * np.sum(dZ1)  #
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1  # updating weights
    b1 = b1 - alpha * db1  # updating weights   
    W2 = W2 - alpha * dW2  # updating weights
    b2 = b2 - alpha * db2  # updating weights
    return W1, b1, W2, b2  # returning weights


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2


W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
