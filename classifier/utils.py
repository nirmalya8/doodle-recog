'''
This file contains the implementation of useful functions for
the task of recognition and model creation. 
'''

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

# Loading the data into numpy arrays

def load_dataset(base_folder, data_split=0.10, data_per_class = 2000): 
    x = np.empty([0,784])
    y = np.empty([0])
    classes = []
    for idx,c in enumerate(os.listdir(base_folder)):
        full_file = os.path.join(base_folder,c)
        data = np.load(full_file)
        data = data[0: data_per_class, :]
        labels = np.full(data.shape[0], idx)
        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)
        class_name, ext = os.path.splitext(os.path.basename(full_file))
        classes.append(class_name)
    
    permutation = np.random.permutation(y.shape[0])
    x = x[permutation, :]
    y = y[permutation]

    data_split = int(x.shape[0]*data_split)
    print(data_split)
    x_test = x[0:data_split,:]
    x_train = x[data_split:,:]

    y_test = y[0:data_split]
    y_train = y[data_split:]
    print(y_test.shape,x_test.shape,y_train.shape,x_train.shape)
        
    return x_train,x_test,y_train,y_test,classes


def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='int')[y]

def vertical_flip(tensor):
    """
    Flips tensor vertically.
    """
    tensor = tensor.flip(1)
    return tensor


def horizontal_flip(tensor):
    """
    Flips tensor horizontally.
    """
    tensor = tensor.flip(2)
    return tensor