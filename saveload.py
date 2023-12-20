import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
 

model = Model(n_input_features=6)

file = "model.pth"

# torch.save(model, file)

# model = torch.load(file)

# model.eval()

# for param in model.parameters():
#   print(param)

# another way to save the model
torch.save(model.state_dict(), file)

# better way to load the model
loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(file))
loaded_model.eval()

# for param in loaded_model.parameters():
#   print(param)

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print('model state dict', model.state_dict())
print('optimizer state dict', optimizer.state_dict())

# checkpoint = {
    # "epoch": 90,
    # "model_state": model.state_dict(),
    # "optim_state": optimizer.state_dict()
# }

# torch.save(checkpoint, 'checkpoint.pth')

# loading the checkpoint
loaded_cp = torch.load('checkpoint.pth')

epoch = loaded_cp['epoch']
model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)

model.load_state_dict(loaded_cp['model_state'])
optimizer.load_state_dict(loaded_cp['optim_state'])

print('from_cp', optimizer.state_dict())

PATH='checkpoint.pth'
# save on CPU, load on GPU
torch.save(model.state_dict(), PATH)

device = torch.device("cuda")
# model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))
model.to(device)
