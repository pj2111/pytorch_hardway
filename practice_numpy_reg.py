# Get the data
# Build the model
# Train the model
    # Forward Pass : predicting train values
    # Backward pass : calculating the gradients and parameters
    # updating weights
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# load datasets
wine_data = datasets.load_wine()
x, y = wine_data['data'], wine_data['target']
# print(type(x))
# print(y.shape)
n_samples, n_features = X.shape

# Split the datasets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=1579)

# Normalize the feature datasets
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

