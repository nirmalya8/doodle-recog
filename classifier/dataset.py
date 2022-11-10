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
from PIL import Image
from utils import load_dataset


class DoodleDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __getitem__(self,idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform is not None:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)

