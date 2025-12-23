import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import torchvision


def plot(train_loss, valid_loss, train_acc, valid_acc):
    # plot loss curves
    plt.plot(range(len(train_loss)), train_loss, label="training")
    plt.plot(range(len(valid_loss)), valid_loss, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("average loss")
    plt.xlim(0, len(train_loss) - 1)
    plt.ylim(0, None)
    plt.legend()
    plt.grid()
    plt.show()

    # plot accuracy curves
    plt.plot(range(len(train_acc)), train_acc, label="training")
    plt.plot(range(len(valid_acc)), valid_acc, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.xlim(0, len(train_acc) - 1)
    plt.ylim(0, None)
    plt.legend()
    plt.grid()
    plt.show()


class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index])
        y = torch.Tensor(self.y[index])
        return (x, y)

    def __len__(self):
        count = self.x.shape[0]
        return count


class CNN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self,x):
        out = self.cnn_model(x)
        return out


train_data = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# model and loss, optimizer
model = CNN_Net()
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# load data, shuffle is necessary!!!
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = DataLoader(test_data, batch_size=64)


epoch = 50
train_loss = []
valid_loss = []
train_acc = []
valid_acc = []
for i in range(epoch):
    # training part
    print("Training iteration: " + str(i+1))
    size = len(train_loader.dataset)
    num_batches = len(train_loader)
    model.train()
    loss_val, correct = 0, 0
    for data in train_loader:
        x, y = data
        pred = model(x)
        loss = loss_fn(pred, y)
        #pred = nn.Softmax(dim=1)(pred)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss_val+=loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_val /= num_batches
    correct /= size
    train_loss.append(loss_val)
    train_acc.append(correct)
    print("Train loss: " + str(loss_val) + ", Accuracy: " + str(correct * 100) + "%")


    # validation part
    size = len(valid_loader.dataset)
    num_batches = len(valid_loader)
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for X, y in valid_loader:
            pred = model(X)
            loss += loss_fn(pred, y).item()
            #pred = nn.Softmax(dim=1)(pred)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    valid_loss.append(loss)
    valid_acc.append(correct)
    print("Validation loss: "+str(loss)+", Accuracy: " + str(correct*100)+"%")

plot(train_loss, valid_loss, train_acc, valid_acc)


