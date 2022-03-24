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

from utils import load_dataset

print(y_train.shape,y_test.shape)
y_train = to_categorical(y_train.astype(int),len(classes))
y_test = to_categorical(y_test.astype(int),len(classes))
print(y_train.shape,y_test.shape)