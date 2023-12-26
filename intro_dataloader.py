import torch
import numpy as np
import torchvision
import math
from torch.utils.data import Dataset, DataLoader

# Script will load the data from text file into a Dataset Class, and 
# use Dataloader, batch processing to create the model and test it

# Before implementing data extraction, review the data. Visualise how the data 
# is required by the dataloader, and plan according to the spec.
# test_wine = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
# print(test_wine[:, [0]])


class WineDataset(Dataset):
    def __init__(self):
        # data reading and loading
        xb = np.loadtxt("wine.csv", delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xb[:, 1:])  # all rows and data from cols 1 till end
        self.y = torch.from_numpy(xb[:, [0]])  # all rows and data in col 0, placed inside list
        self.n_samples = xb.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
 
    def __len__(self):
        return self.n_samples


winedata = WineDataset()
features, target = winedata[0]
# print(features, target)
# implement Dataloader
wineloader = DataLoader(dataset=winedata,
                        batch_size=4,
                        shuffle=True,)
wineiter = iter(wineloader)
# data = wineiter.next()  # raises the '_MultiProcessingDataLoaderIter' object has no attribute 'next'
# https://stackoverflow.com/questions/74289077/attributeerror-multiprocessingdataloaderiter-object-has-no-attribute-next
data = next(wineiter)
# RuntimeError: DataLoader worker (pid(s) 8152) exited unexpectedly
# https://stackoverflow.com/questions/60101168/pytorch-runtimeerror-dataloader-worker-pids-15332-exited-unexpectedly 
# solution is to remove the num_workers 
# another solution is mod the builder.py (https://github.com/open-mmlab/mmsegmentation/issues/1482)
# Set cfg.data.workers_per_gpu=0
# Open mmseg>datasets>builder.py
# Change Line 99 as persistent_workers=False
feats, targets = data
# print(feats, targets)

# training loop
num_epochs = 2  # entire data will be sent through the forward pass and then gradients calculated
total_samples = len(winedata)
n_iteration = math.ceil(total_samples / 4)  # There will 10 iters if there are 40 datapoints

# print(total_samples, n_iteration)

for epoch in range(num_epochs):
    # when you are enumerating the data & updating the weights 
    # DataLoader object has to be used. Not the Data Iterator
    for i, (inputs, labels) in enumerate(wineloader):
        # forward backward and update
        if (i + 1) % 5 == 0:
            print(f"epoch: {epoch} / {num_epochs}, step: {i+1} / {n_iteration}, inputs: {inputs.shape}")
        # the training and testing part of the script can be updated below