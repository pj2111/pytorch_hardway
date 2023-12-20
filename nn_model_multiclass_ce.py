# introduces how cross entropy loss is used with the Neural 
# network layers
import torch
import torch.nn as nn


class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)  # observe there is no SoftMax layer in the forward method.
        return out

model = NeuralNet2(input_size=28*28, hidden_size=6, num_classes=3)
criterion = nn.CrossEntropyLoss()