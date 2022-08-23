'''
This file will be used to declare the various layers of the network
and then the training loop, using which it will be trained on the
dataloader created from the DoodleDataset object. 
'''

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def change_layers(model):
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(2048, 10, bias=True)
    return model 


def get_model():
    model = models.resnet50(pretrained=True)
    model = change_layers(model)
    return model
