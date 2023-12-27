import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
# print(device)
# zeros, ones, rand, empty, randint, linspace, eye, 
# multinomial, cat, arange, unsqueeze(learn), masked_fill 
# stack, triu, tril, transpose, softmax, 
# Embedding, Linear functions, data loading
# view, broadcasting semantics, transforms
emb = nn.Embedding(num_embeddings=13, embedding_dim=3)
# print(emb.weight)
input = torch.LongTensor([13, 8, 5, 0])
# input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9], [4, 3, 2, 9], [4, 3, 2, 9]])
# print(emb(input))

# working on broadcasting semantics
x = torch.empty(5, 6, 3)
y = torch.empty(5, 6)  # they are not broadcastable
r = torch.empty(5, 6, 1)  # Broadcastable with x

# z = x @ r

# print(z)

tensor1= torch.rand(3)  # 1 * 3 cols
tensor2= torch.rand(3)  # 1 * 3 cols

# print(tensor1)
# print(tensor2)

tm1 = torch.matmul(tensor1, tensor2)
# print(tm1)

tensor3 = torch.rand(3, 4)  # 3 * 4 cols
tensor4 = torch.rand(4)  # 4 cols

# tm2 = torch.matmul(tensor4, tensor3)
tm2 = torch.matmul(tensor3, tensor4)
# print(tm2)


# batched matrix with broadcasted vector
tensor5 = torch.rand(10, 3, 4)
tensor6 = torch.rand(4)

tm3 = torch.matmul(tensor5, tensor6)
# print(tm3)

# batched matrix with broadcasted matrix
tensor9 = torch.rand(10, 3, 4)
tensorh = torch.rand(4, 5)

mt6 = torch.matmul(tensor9, tensorh)
print(mt6)

