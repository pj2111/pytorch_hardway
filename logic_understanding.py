import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

block_size = 4
batch_size = 6

a = torch.randint(100 - block_size, (batch_size, ))  # argument 'size' (position 2) must be tuple of ints, not int
# print(a.shape)
# print(a)

