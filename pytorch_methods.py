import torch
import torch.nn as nn
import numpy as np
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)

# zeros
a = torch.zeros(1, 2)  # 1 row and 2 cols 
# print(a)

# ones
b = torch.ones(2, 2)  # 2 row and 2 cols
# print(b)

# rand
c = torch.rand(10, 10)  # 10 rows and 10 cols
# print(c)

# empty
d = torch.empty(5, 3)  # 5 rows and 3 cols
# print(d)

# randint
e = torch.randint(low=10, high=100,
                  size=(3, 2))  # 3 row and 2 cols random number between 10 and 100
# print(e)

# linspace
f = torch.linspace(start=0, end=13, steps=4, dtype=torch.int32)  # linear space between 0 to 7 with steps of 3
# can control the type of the output without worrying about type error
# print(f)

# eye or identity matrix
g = torch.eye(n=5, m=5)
# print(g)

# following are methods that check the efficiency or provides some easy way of manipulating the data

h = torch.rand(size=[100, 100, 100, 100])
# print(h.shape)

i = torch.rand(size=[100, 100, 100, 100])

# do torch multiplication
tm = (h @ i)
# print(h[0][0][0][0], 'h')
# print(i[0][0][0][0], 'i')
# print(tm[0][0][0][0], 'hm')
# do numpy multiplication
nm = np.multiply(h, i)
# print(nm[0][0][0][0], 'nm')

# embeddings, torch.stack, torch.multinomial, torch.tril, torch.triu
# input.T / input.transpose, nn.Linear, torch.cat, F.softmax
# (show all the examples of functions/methods with pytorch docs)
j = torch.tensor([0.2, 0.3, 0.5])
samples = torch.multinomial(input=j, num_samples=10, replacement=True)
# draws samples with 0, 1, 2 
# print(samples)

k = torch.tensor([1, 2, 3, 4, 5])
l = torch.tensor([5])  # torch.Size([1])
# print(l.shape)
m = torch.tensor(5)  # this will have no size torch.Size([])
# print(m.shape)

o = torch.cat((k, l),dim=0)
# print(o)
# n = torch.cat((k, m), dim=0)  # will throw zero-dimensional tensor cannot be concatenated
# print(n)

p = torch.randint(low=5, high=10, size=(2, 3))
q = torch.randint(low=10, high=25, size=(2, 4))
# print(p)
# print(q)

r = torch.cat((p, q), dim=1)  # tensor 1 is in 2nd position, dim = 0 / 1 will work on two axes only
# print(r)

# stack expects each tensor to be equal size, but got [5] at entry 0 and [4] at entry 3
s1 = torch.arange(0, 5)
s2 = torch.arange(1, 6)
s3 = torch.arange(2, 7)
s4 = torch.arange(4, 9)
s5 = torch.stack((s1, s2, s3, s4))
# print(s5.shape)
# print(s5)

tl = torch.tril(torch.ones(3, 3) * 5)  # scalar int multiplication works
# print(tl)
tu = torch.triu(torch.ones(3, 3) * torch.tensor([5]))  # What happens when 
# another tensor is involved? Same result
# print(tu)

tu_try = torch.triu(torch.ones(3, 3) * torch.tensor(5))  # What happens when 
# another tensor with None size is involved? 
# print(tu_try)

maskout = torch.zeros(5, 5).masked_fill(torch.tril(torch.ones(5, 5)) == 0,
                                        float('-inf'))
# print(maskout, 'maskout')
# print(torch.exp(maskout), 'exponentiating maskout')

# print(torch.exp(torch.tensor([0])), 'mask out')
# print(torch.exp(torch.tensor([float('-inf')])), 'mask out')

input = torch.zeros(2, 3, 4)
# print(input.shape, 'input')

out1 = input.transpose(0, 1)
# help(torch.transpose)
out2 = input.transpose(-2, -1)
# The resulting :attr:`out`
# tensor shares its underlying storage with the :attr:`input` tensor, so
# changing the content of one would change the content of the other.
# LOOK AT THE SHAPE...

# print(out1.shape, 'out1')
# print(out2.shape, 'out2')

# How the linear works?

import torch.nn.functional as F

ten1 = torch.tensor([1., 2., 3.])
# print(type(ten1))
# the tensor is int64 type by default, need to make it float by adding a '.' point
lin1 = nn.Linear(3, 1, bias=False)
lin2 = nn.Linear(1, 1, bias=False)
# print(lin1(ten1))
# print(lin2(ten1))  # will error out as the dims don't match
# mat1 and mat2 shapes cannot be multiplied (1x3 and 1x1)

# How softmax works?
s_out = F.softmax(ten1)
print(s_out)

# How embedding works 
vocab_size = 80
embedding_dim = 6

r_in = nn.Embedding(num_embeddings=vocab_size, 
                     embedding_dim=embedding_dim)
data_ind = torch.tensor([1, 5, 6, 8])
e_out = r_in(data_ind)
print(e_out)
print(e_out.shape)
"""
tensor([[-0.5251, -2.2980, -1.2629, -0.2184, -0.3236, -1.1250],
        [-2.0372, -0.7762,  1.1529, -1.7969,  0.3080, -0.4566],
        [ 0.3185,  1.7108, -0.4360,  1.5348, -1.1450,  0.2744],
        [-0.0502, -1.8797,  1.3616, -0.0599, -0.4435,  0.0271]],
       grad_fn=<EmbeddingBackward0>)
"""
# Matrix Multiplication
a = torch.tensor([[1, 2], [3, 4], [5, 6]])
b = torch.tensor([[7, 2, 9], [6, 3, 4]])

# print(a @ b)
# print(a.matmul(b))
# print(torch.matmul(a, b))

# playing with shapes

input = torch.rand((3, 8, 10))
B, T, C = input.shape
output = input.view(B * T, C)
# print(output.shape)
# print(output[:2, :-1])

b = torch.tensor([[1, 72, 68, 74, 67, 57, 72,  0],
                  [74, 73, 58, 71,  0, 56, 54, 75]], dtype=torch.float32)

d = torch.tensor([[1, 72, 68, 74, 67, 57, 72,  0, 73],
                  [74, 73, 58, 71,  0, 56, 54, 75, 58]], dtype=torch.float32)

# print(b.shape)
# print(d.shape)

print(b.view(2 * 8))
print(d.view(2 * 9))

ce = F.cross_entropy(b.view(2*8), d.view(2*9))
print(ce)
