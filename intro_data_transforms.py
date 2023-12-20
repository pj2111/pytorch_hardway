import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import numpy as np
import math
# more about transforms can be learnt at https://pytorch.org/vision/stable/transforms.html#transform-classes-functionals-and-kernels
data = torchvision.datasets.MNIST(
    root='./data', transform=torchvision.transforms.ToTensor(),
    download=False
)  # the dataset can be downloaded by passing download = True
# print(data[0])


class Winedataset(Dataset):
    def __init__(self, transform=None):
        self.xy = np.loadtxt('wine.csv',
                             dtype=np.float32,
                             skiprows=1,
                             delimiter=',')
        self.x = self.xy[:, 1:]
        self.y = self.xy[:, [0]]
        self.n_samples = self.x.shape[0]
        # if user asks for transform, store it here
        self.transform = transform

    def __getitem__(self, index):
        samples = self.x[index], self.y[index]
        if self.transform:
            samples = self.transform(samples)
        return samples

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        # print(inputs, end='\n')
        # print(targets, end='\n')
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target


winedata = Winedataset(transform=ToTensor())

firstdata = winedata[0]
# print(type(firstdata[0]), type(firstdata[1]))

wine_mulfactor = Winedataset(transform=MulTransform(2))
seconddata = wine_mulfactor[0]
print('trial transform', seconddata[0])

compose = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = Winedataset(transform=compose)

print(dataset[0])

