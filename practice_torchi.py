import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn import datasets


# Practice the methods (did some additional work on understanding the methods)
# Explore sklearn data
# load data into torch
# create data loader and loop through batches

a = torch.zeros(1, 2)  # 1 row 2 cols
#print(a)

b = torch.ones(1, 2)  # 1 row 2 cols
#print(b.shape)

c = torch.rand(2, 5)  # 2 row 5 cols
#print(c)

d = torch.empty(4, 5)  # 4 row 5 cols

e = torch.randint(low=0, high=10, size=(2, 5))
# print(e)

f = torch.linspace(start=5, end=30, steps=10)  # 0 row 10 cols
# print(f.shape)

# can we shape the linspace tensors
fv = f.view(-1, 2)
# print(fv.shape)

# unsqueeze the a tensor
a = torch.zeros(3, 2)  # 3 row 2 cols
a = a.unsqueeze(dim=1)
# print(a)

e = torch.eye(n=5)  # 5 X 5 output

# print(e.shape)

g = torch.tensor([2, 5, 8, 9, 3, 6, 7], dtype=torch.float16)
# print(g.shape)
ug = torch.unsqueeze(g, 0)
# print(ug.shape)
ug1 = torch.unsqueeze(g, 1)
# print(ug1.shape)
# ug2 = torch.unsqueeze(g, 2)  # won't work, throws index error
# print(ug2)

l = g.shape[0]
prob = torch.tensor([0.5, 0.2, 0.3])
# draws the number as per the probability
multn = torch.multinomial(prob, num_samples=l, replacement=True)
# print(multn)

part1 = torch.rand(size=(2, 2))  # 2 row / 2 col matrix
part2 = torch.rand(size=(2, 1))  # 2 row 1 col matrix
part3 = torch.rand(size=(1, 2))  # 1 row 2 col matrix

cat2 = torch.cat((part1, part3))  # will go through 
# print(cat2)
# cat1 = torch.cat((part1, part2))  # will inform the expected dimensions are not present

part4 = torch.rand((4, 3), dtype=torch.float16)
part5 = torch.randint(high=100, low=10, size=(4, 3), dtype=torch.float16)

partstack = torch.stack([part4, part5])

# print(partstack.shape)  # 2 tensors of 4 rows and 3 cols 

# partstackx0 = torch.stack(part4, part5)  # will throw syntax-error 

# print(partstackx0.shape)  # 2 tensors of 4 rows and 3 cols

catparts_dim0 = torch.cat((part4, part5), dim=0)
catparts_dim1 = torch.cat((part4, part5), dim=1)

# print(catparts_dim0.shape)
# print(catparts_dim1.shape)
embed_matrix = torch.tensor([[2, 6, 8, 7, 43],
                             [9, 86, 78, 90, 62]],
                             dtype=torch.float32)
part7 = torch.tensor([1, 2, 5, 6])
part8 = torch.tensor([1, 2, 5, 6], dtype=torch.float32)
em = nn.Embedding(num_embeddings=8,
                  embedding_dim=6)
# print(em(part7))

lin = nn.Linear(in_features=4, out_features=1)
# print(lin(part8))
lin2 = nn.Linear(in_features=4, out_features=2)
# print(lin2(part8))

part9 = torch.linspace(start=10, end=20, steps=30)

# print(part9.shape)

# print(part9.view(size=(10,3)))
# print(part9.view(size=(5,6)))
# print(part9.view(size=(3,10)))
# print(part9.view(size=(4,10)))  # RuntimeError: shape '[4, 10]' is invalid for input of size 30

# print(part9.view(size=(-1, 3)))
# print(part9.view(size=(-1, 4)))  # will be invalid

# working on data loader
regression_ds = datasets.load_wine()
x_data, y_data = regression_ds['data'], regression_ds['target']

# print(x_data.shape)
# print(y_data.shape)

# data is ready
class WineDataset(Dataset):
    # just use the init to load the data into the torch objects
    def __init__(self, x_data, y_data):
        self.x = torch.from_numpy(x_data.astype(np.float32))
        self.y = torch.from_numpy(y_data.astype(np.float32))
        self.y = self.y.view(y_data.shape[0], 1)
        self.n_samples = x_data.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


wine_ds = WineDataset(x_data=x_data, y_data=y_data)
winedataloader = DataLoader(wine_ds, shuffle=True, batch_size=3)

batch1 = next(iter(winedataloader))
# print(batch1)

# parameters
n_features, n_samples = wine_ds.x.shape[1], len(wine_ds)
learn_rate = 0.01
epoch = 5

# simple model is ready
class Lnreg(nn.Module):
   
    def __init__(self, n_inputs):
        super().__init__()
        self.lin = nn.Linear(n_inputs, 1)
 
    def forward(self, x):
        y_pred = torch.sigmoid(self.lin(x))
        return y_pred


model = Lnreg(n_inputs=n_features)
# create the criterion
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

# trackers 
running_loss = 0.0
running_corrects = 0
# training loop
for ep in range(epoch):
    # send all the data into the model 
    for ind, batch in enumerate(iter(winedataloader)):
        # print(f"working on the {ind + 1} batch")
        # print(type(batch[0].dtype))
        outputs = model(batch[0])
        # print(outputs)
        # predict after the model weights training
        # print(batch[1])
        loss = criterion(outputs, batch[1])

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"loss output of epoch {ep} is {loss.item():.3f}")

with torch.no_grad():
    y_pred = model(torch.tensor(x_data.astype(np.float32)))
    print(y_pred)