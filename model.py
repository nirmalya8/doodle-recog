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

x_train,x_test,y_train,y_test,classes = load_dataset('Data')

# def to_categorical(y, num_classes):
#     n = y.shape[0]
#     input_shape = y.shape
#     input_shape = y.shape
#     if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
#         input_shape = tuple(input_shape[:-1])
#     y = y.ravel()
#     if not num_classes:
#         num_classes = np.max(y) + 1
#     categorical = np.zeros((n, num_classes), dtype='int')
#     categorical[np.arange(n,dtype='int') , y] = 1
#     output_shape = input_shape + (num_classes,)
#     categorical = np.reshape(categorical, output_shape)
#     return categorical
#    
def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='int')[y]


print(y_train.shape,y_test.shape)
y_train = to_categorical(y_train.astype(int),len(classes))
y_test = to_categorical(y_test.astype(int),len(classes))
print(y_train.shape,y_test.shape)