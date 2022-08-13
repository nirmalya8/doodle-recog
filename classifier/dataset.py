"""
This file is used to generate a Custom Dataset object by 
creating a class that takes in the x_train and y_train from the
load_dataset function and creates a custom dataset from it. It can
then be used to create a dataloader, on which the model will be trained.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

from utils import load_dataset


class DoodleDataset(Dataset):
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self,idx):
        x = self.tensors[0][idx]
        if self.transform:
            x = self.transform(x)
        y = self.tensors[1][idx]
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

