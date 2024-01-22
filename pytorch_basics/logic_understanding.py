import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

block_size = 4
batch_size = 6

a = torch.randint(100 - block_size, (batch_size, ))  # argument 'size' (position 2) must be tuple of ints, not int
# print(a.shape)
# print(a)

b = torch.tensor([[1, 72, 68, 74, 67, 57, 72,  0],
                  [74, 73, 58, 71,  0, 56, 54, 75]])

d = torch.tensor([[1, 72, 68, 74, 67, 57, 72,  0, 73],
                  [74, 73, 58, 71,  0, 56, 54, 75, 58]])

