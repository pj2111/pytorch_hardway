import numpy as np
from sklearn import datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

device = 'cpu' if not torch.cuda.is_available() else 'cuda'

a = torch.zeros(3, 2)
# print(a.shape)

asq = torch.squeeze(a, dim=0)  # expected row dimension to be removed...
# print(asq)  # did not happen

asu = torch.unsqueeze(a, dim=0)
# print(asu.shape)

b = torch.ones(3, 2)
# print(b.shape)

c = torch.rand(3, 2)
# print(c.shape)

# print((a + b + c).shape)

d = torch.randint(low=5, high=68, size=(4, 3, 2))
# print(d)

e = d.view(-1, 2)  # 12 r and 2 c
# print(e)

f = d.view(2, -1)  # 2 r and 12 c
# print(f)

# g = d.view(3, -1, -1) 

g = d.view(3, -1, 2)
# print(g.shape)  # 3 mat of 3 r and 3 c
# print(g[0].T)  # 2 r and 4 c

i = torch.arange(3, 28, 3)  # how many elements?
j = torch.arange(4, 29, 3)  # how many elements?
k = torch.arange(5, 30, 3)  # how many elements?

# print(torch.stack((i, j, k)))

# bring in the datasets
regress = datasets.make_regression(n_samples=50,
                                   n_features=5,
                                   n_targets=1,
                                   random_state=123)

# print(regress)

# load into torch

x_data, y_data = regress

x_torch = torch.from_numpy(x_data.astype(np.float32))
y_torch = torch.from_numpy(y_data.astype(np.float32))

# print(x_torch.shape)  # 50, 5
# print(y_torch.shape)  # 50, 2

# break it into batches
x_batch = x_torch.view(10, 5, 5)
y_batch = y_torch.view(10, 5, -1)
# print(y_batch.shape)
# print(x_batch.shape)

# print(y_batch[0])

# print(datasets.get_data_home())

# mul_x_y = x_batch[0] * y_batch[0]  # will error out

# print(mul_x_y)
# what should be the weight shape? unable to visualise 
# as there is no practice 

linear = nn.Linear(in_features=5, out_features=1)
# print(linear.weight.shape)  # shape of 2r 5c 
# print(linear.weight)  # shape of 2r 5c 

# how feed the data to model

batch_lin0 = linear(x_batch[0])

# print(batch_lin0.shape)  # expecting 5r 2c output

sx = F.softmax(batch_lin0, dim=0)

# print(sx.shape)
# print(sx)

criterion = torch.nn.MSELoss()

# loss = criterion(sx, y_batch[0])

# print(loss)
# loss.backward()

optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
# optimizer.step()
# optimizer.zero_grad()
# print(linear.weight)

# epoch = 50
# for epo in range(epoch):
    # for ind, bt in enumerate(x_batch):
        # y_p_bt = linear(bt)
        # loss_bt = criterion(y_p_bt, y_batch[ind])
        # loss_bt.backward()
        # optimizer.step()
        # optimizer.zero_grad()
    # print(f"loss: {loss_bt.item():.3f}")


data = datasets.load_digits()
# print(data.keys())
data = data['data']
# target = data['target']


class ToTensor:
    def __call__(self, data_points):
        print(data_points)
        data = data_points[0]
        target = data_points[1]
        return torch.from_numpy(data.astype(np.float32)), torch.tensor(target, 
                                                                       dtype=torch.float32)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, samples):
        inputs, targets = samples
        inputs *= self.factor
        return inputs, targets


class DigitDataset(Dataset):
    def __init__(self, transform=None) -> None:
        self.data_set = datasets.load_digits()
        self.data = self.data_set['data']
        self.target = self.data_set['target']
        self.n_samples = self.data.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.data[index], self.target[index]
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples

dd = DigitDataset(transform=ToTensor())

print(dd[0])