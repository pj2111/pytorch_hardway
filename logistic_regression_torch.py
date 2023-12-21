# 1) Designing the model
# 2) Assigning the Loss Function & optimizer
# 3) Training Loop
#   - Forward Pass : Predicting value
#   - Backward Pass: Predicting gradient & optimizing params
#   - Updating weights

import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

#0 prep the data
given = datasets.load_breast_cancer()
X, y = given.data, given.target
print(X[:2])  # list of lists
print(y[:2])  # list of 0 and 1
print(X.shape)  # (569, 30)
n_samples, n_features = X.shape  # refer above line for the values assigned
# can the train_test_split work on torch arrays?
x_torch = torch.from_numpy(X.astype(np.float32))
y_torch = torch.from_numpy(y.astype(np.float32))
# split the datasets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=1)
# scale the inputs, to have 0 mean and unit variance
sc = StandardScaler()
# use the train set to create the scaler model
X_train = sc.fit_transform(X_train)
# use the fit model to transform the test set
X_test = sc.transform(X_test)
# make tensors from numpy arrays
X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
# reshape the target tensors
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# 1 Build model from scratch
class LogRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogRegression, self).__init__()
        # there are n features and output is a single number 0 / 1
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted


model = LogRegression(n_features)

# loss and optimizer
lr = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# train loop
epochs = 100

for epo in range(epochs):
    # forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epo % 10 == 0:
        print(f"epoch: {epo}, loss: {loss.item():.3f}")

# lets do the evaluation
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_cls = y_pred.round()
    # check how many predicted are matching y_test and sum it
    # divide it by number of y_test elements
    acc = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f"accuracy = {acc:.3f}")
