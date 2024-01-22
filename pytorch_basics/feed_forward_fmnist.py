import torch
from torch import nn
from torch import functional as F
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
num_epochs = 5
num_features = 784
batch_size = 10
learning_rate = 0.01
hidden_size = 100
output_size = 10

train_data = torchvision.datasets.FashionMNIST(
    root='./data',train=True,transform=transforms.ToTensor(),download=False
)
test_data = torchvision.datasets.FashionMNIST(
    root='./data',train=False,download=False,transform=transforms.ToTensor()
)

# sample, label = train_data[0]

# print(sample.shape)  # size([1, 28, 28])

# print(label.shape)  # 

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# print(len(train_loader))

class MnistModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(MnistModel, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)
        return out 

model = MnistModel(num_features, hidden_size, output_size)
model.to(device)
criteria = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(params=model.parameters(),
                             lr=learning_rate,)
running_loss = 0
running_correct = 0

for ep in range(num_epochs):
    print(f"Epoch is: {ep}")
    for i, (sample, label) in enumerate(train_loader):
        sample = sample.reshape(-1, 28 * 28).to(device)
        label = label.to(device)
        pred = model(sample)
        loss = criteria(pred, label)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        running_loss += loss.item()
        _, pred_classes = torch.max(pred.data, 1)
        running_correct += (pred_classes == label).sum().item()
    print(f"Epoch is: {ep} / Loss is: {running_loss} / corrects are: {running_correct}")

smax = nn.Softmax(dim=0)
correct_pred = 0

# for i, (test, label) in enumerate(test_loader):
    # # print(label.shape)
    # model.eval()
    # with torch.no_grad():
        # pred = model(test.view(-1, 28 * 28))
        # pred_lable = smax(pred)
        # # print(len(pred_lable))
        # for ind, lab in enumerate(pred_lable):
            # if lab == label[ind]:
                # correct_pred += 1
smax = nn.Softmax(dim=0)
test_batch, test_label = next(iter(test_loader))
test_batch = test_batch.reshape(-1, 28 * 28).to(device)
test_label = test_label.to(device)
with torch.no_grad():
    label_pred = model(test_batch)
    pred_smax = smax(label_pred)
    _, pred_class = torch.max(pred_smax, 1)
    correct_pred += (test_label == pred_class).sum().item()

print(f"Correct values are : {correct_pred}")
# print(f"Accuracy is : {correct_pred / len(test_loader)}")