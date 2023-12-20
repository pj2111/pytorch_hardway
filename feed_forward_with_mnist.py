import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn import functional as F
# working on the tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='runs/mnist')
# MNIST data loader
# Dataload and transformation
# Design Multi-layer neural net with activation function
# Loss and Optimiser declaration
# Training loop with batches
# model evaluation using accuracy
# enable GPU support

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# preliminary data
input_size = 784
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# Getting data 
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                           transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                          transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                         shuffle=False)

examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)  # torch.Size([100, 1, 28, 28]) torch.Size([100])

for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(samples[i][0], cmap='gray')

# plt.show()
img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist images', img_grid)
writer.close()
# sys.exit()

# Model definition
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size=input_size,
                  hidden_size=hidden_size,
                  num_classes=num_classes)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
writer.add_graph(model=model, input_to_model=samples.reshape(-1, 28 * 28))
writer.close()
# sys.exit()
n_total_steps = len(train_loader)
running_loss = 0.0
running_corrects = 0

for epoch in range(num_epochs):
    print(f"entering epoch: {epoch + 1}\n")
    for i, (images, labels) in enumerate(train_loader):
        # 100, 1, 28, 28
        # 100, 784
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

    # forward pass 
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        running_corrects += (predicted == labels).sum().item()
        if (i+1) % 100 == 0:
            print(f"epoch: {epoch+1} / {num_epochs}, step {i + 1} / {n_total_steps} loss= {loss.item():.3f}")
            writer.add_scalar('training_loss', running_loss / 100, epoch * n_total_steps)
            writer.add_scalar('training_corrects', running_corrects / 100, epoch * n_total_steps)
            running_loss = 0.0
            running_corrects = 0
    # writer.close()
    # sys.exit()
labels = []
predictions = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, samples in test_loader:
        # convert the images to 1D tensor, and send to device 
        images = images.reshape(-1, 28 * 28).to(device)
        # send labels to device
        labels = labels.to(device)
        # predict outputs
        outputs = model(images)
        print(outputs.shape)
        _, prediction = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (prediction == labels).sum().item()
        # following used for tensor board, accuracy and recall calculation
        class_preds = [F.softmax(output, dim=0) for output in outputs]
        labels.append(prediction)
        predictions.append(class_preds)
    preds = torch.cat([torch.stack(batch) for batch in predictions])
    labels = torch.cat(labels)
    acc = 100 * (n_correct / n_samples)

    print(f"accuracy is {acc}")

    classes = range(10)
    for i in classes:
        labels_i = labels == i
        preds_i = preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i)