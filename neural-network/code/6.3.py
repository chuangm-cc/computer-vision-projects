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
input_size = 224

data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

val_data=torchvision.datasets.ImageNet(root="D:\CV", split='val', transform=data_transforms)
val_data = DataLoader(val_data, batch_size=10)

# model and loss, optimizer
model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
model.eval()
k = 0
acc=0
sum=0
# tench and goldfish
with torch.no_grad():
    for X, y in val_data:
        if int(y[0])!=k:
            # accurancy
            res = (acc/sum)*100
            print("Accurancy for category: "+str(k) +" is "+str(res)+"%")
            # next category
            k+=1
            sum=0
            acc=0
        # only get accurancy for first 4
        if k==4:
            break
        pred = model(X)
        acc+=(pred.argmax(1) == y).type(torch.float).sum().item()
        sum+=10


