import torch
import torchvision
from torchvision import transforms, datasets
import os.path

import skimage
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet152,ResNet152_Weights
img_size = 260
input_size = 224

data_transforms = transforms.Compose([
    #transforms.Resize((img_size, img_size)),
    #transforms.CenterCrop((input_size, input_size)),
    transforms.RandomResizedCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# car
real = 681
hymenoptera_dataset = datasets.ImageFolder(root='../data/real_data',
                                           transform=data_transforms)
val_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                             batch_size=10, shuffle=True)

# model and loss, optimizer
model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
model.eval()
acc=0
sum=0
with torch.no_grad():
    for X, y in val_loader:
        pred = model(X)
        acc+=(pred.argmax(1) == real).type(torch.float).sum().item()
        sum+=10
    res = (acc / sum) * 100
    print("Accurancy for category: "  + str(real) + " is " + str(res) + "%" +" ("+str(acc) +"/"+str(sum)+")")


