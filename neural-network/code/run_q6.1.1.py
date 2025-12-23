import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt


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


class FC_Net(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.fc1 = nn.Linear(size_in,64)
        self.fc2 = nn.Linear(64, size_out)

    def forward(self,x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        out = self.fc2(x)
        return out


train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

# model and loss, optimizer
model = FC_Net(train_x.shape[1], train_y.shape[1])
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# load data, shuffle is necessary!!!
train_data = Dataset(train_x,train_y)
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

test_data = Dataset(test_x,test_y)
test_loader = DataLoader(test_data, batch_size=100)

valid_data = Dataset(valid_x,valid_y)
valid_loader = DataLoader(valid_data, batch_size=100)

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
        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
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
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    valid_loss.append(loss)
    valid_acc.append(correct)
    print("Validation loss: "+str(loss)+", Accuracy: " + str(correct*100)+"%")

plot(train_loss, valid_loss, train_acc, valid_acc)


