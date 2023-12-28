import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
from sklearn import datasets
# multinomial, cat, arange, unsqueeze(learn) 
# stack, softmax, 
# Embedding, Linear functions, data loading
# view, broadcasting semantics, transforms

# get a 3 X 3 dim data set


class Regdataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.X = torch.from_numpy(data[0].astype(np.float32))
        self.Y = torch.from_numpy(data[1].astype(np.float32))
        self.y = self.Y.view(self.X.shape[0], -1)
        self.samples = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.samples


data1 = datasets.make_regression(n_samples=3,
                                 n_features=3,
                                 noise=3,
                                 random_state=3)
data1 = Regdataset(data1)
data2 = datasets.make_regression(n_samples=3,
                                 n_features=3,
                                 noise=37,
                                 random_state=3675)
data2 = Regdataset(data2)
data3 = datasets.make_regression(n_samples=3,
                                 n_features=3,
                                 noise=367,
                                 random_state=7863)
data3 = Regdataset(data3)

# vector * vector, matrix * vector, batched mat * vector(bc) 
# batched matrix * batched matrix, batched matrix * broadcasted matrix
# A vector is of shape 1 row n col


vector = torch.tensor([5, 6, 8])
# print(vector.shape)

tensor5 = torch.rand(3, 4, 3)
tensor_b = torch.tensor([[[1, 2, 4, 5], [5, 6, 8, 9], [5, 6, 4, 2]],
                         [[1, 9, 0, 8], [7, 2, 4, 5], [9, 8, 6, 2]],
                         [[7, 6, 2, 3], [7, 6, 3, 5], [6, 7, 3, 4]]
                         ], dtype=torch.float32)
print(tensor_b.shape)
print(tensor5.shape)
print(tensor_b @ tensor5)
# matrix = torch.tensor([[1, 2, 4],
                    # [5, 6, 8],
                    # [9, 6, 5]])

# # v X v is acceptable

# v1 = vector @ matrix 
# v8 = matrix @ vector   
# print(v8)
# v2 = torch.matmul(matrix, vector)
# print(v2)
# # v X m is acceptable


