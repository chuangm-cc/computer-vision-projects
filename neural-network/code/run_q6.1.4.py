import os.path

import skimage
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import numpy as np



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

img_size = 64

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
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )

    def forward(self,x):
        out = self.cnn_model(x)
        return out


def get_data(path,x_txt,y_txt):
    # for img
    f = open(x_txt,'r')
    lines = f.readlines()
    #print(lines)
    f.close()
    x = []
    for line in lines:
        line = line.replace("\n", "")
        img_path = path+line
        img = Image.open(img_path)
        img = img.resize((img_size, img_size))
        img = np.array(img.convert('L'))
        img=img/255.0
        x.append(img)

    f = open(y_txt, 'r')
    lines = f.readlines()
    f.close()
    y = []
    for line in lines:
        line = line.replace("\n", "")
        y.append(int(line))

    y = np.array(y)
    # print(y)
    x = np.array(x)
    # N = x.shape[0]
    # # N,channel,h,w
    # x = x.reshape(N, 1, 64, 64)
    y=y.reshape(-1,1)
    # print(y)
    print(x.shape)
    print(y.shape)
    return x,y


# read
train_x_path = '../data/SUN/train_files.txt'
train_y_path = '../data/SUN/train_labels.txt'
test_x_path = '../data/SUN/test_files.txt'
test_y_path = '../data/SUN/test_labels.txt'
path = '../data/SUN/'
# save
x_train_save = '../data/sun_train_x.npy'
y_train_save = '../data/sun_train_y.npy'
x_test_save = '../data/sun_test_x.npy'
y_test_save = '../data/sun_test_y.npy'

if os.path.exists(x_train_save):
    train_x = np.load(x_train_save)
    train_y = np.load(y_train_save)
    valid_x = np.load(x_test_save)
    valid_y = np.load(y_test_save)
else:
    train_x, train_y = get_data(path, train_x_path, train_y_path)
    valid_x, valid_y = get_data(path, test_x_path, test_y_path)
    np.save(x_train_save,train_x)
    np.save(y_train_save, train_y)
    np.save(x_test_save, valid_x)
    np.save(y_test_save, valid_y)

# model and loss, optimizer
model = CNN_Net()
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# load data, shuffle is necessary!!!
train_data = Dataset(train_x,train_y)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

valid_data = Dataset(valid_x,valid_y)
valid_loader = DataLoader(valid_data, batch_size=32)

epoch = 30
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
        N = x.shape[0]
        # N,channel,h,w
        x = x.reshape(N,1,img_size,img_size)
        y=y.reshape(-1)
        #print(y.dtype)
        pred = model(x)
        y = y.long()

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
            N = X.shape[0]
            # N,channel,h,w
            X = X.reshape(N, 1, img_size, img_size)
            pred = model(X)
            y = y.reshape(-1)
            y = y.long()
            loss += loss_fn(pred, y).item()
            #pred = nn.Softmax(dim=1)(pred)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    valid_loss.append(loss)
    valid_acc.append(correct)
    print("Validation loss: "+str(loss)+", Accuracy: " + str(correct*100)+"%")

plot(train_loss, valid_loss, train_acc, valid_acc)


