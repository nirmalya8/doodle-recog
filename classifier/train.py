'''
This file will be used to declare the various layers of the network
and then the training loop, using which it will be trained on the
dataloader created from the DoodleDataset object. 
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import time
from utils import load_dataset,to_categorical,horizontal_flip, vertical_flip
from dataset import DoodleDataset
from model import LeNet 
from torchvision import models

image_size = 28
x_train,x_test,y_train,y_test,classes = load_dataset('./Data')
x_train = x_train.reshape(x_train.shape[0], image_size, image_size, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], image_size, image_size, 1).astype('float32')
print(classes)
transforms = transforms.Compose([transforms.Resize((32, 32)),vertical_flip])
x_train /= 255.0
x_test /= 255.0


y_train = to_categorical(y_train.astype(int),len(classes))
y_test = to_categorical(y_test.astype(int),len(classes))

x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)

train_dataset = DoodleDataset(tensors=(x_train, y_train), transform=transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)

x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test)

test_dataset = DoodleDataset(tensors=(x_test, y_test), transform=transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

print(y_train.shape,y_test.shape)
print(x_train.shape,x_test.shape)

model = LeNet()
#if(torch.cuda.is_available()):
#    model = model.cuda()

print('Training....')
total = 0
correct = 0
start = time.time()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 10
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        #images = images.to(device)
       # labels = labels.to(device)
        
        #Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        	
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        		
        if (i+1) % 400 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

print('Training Completed in: {} secs'.format(time.time()-start))
print('Training accuracy: {} %'.format((correct/total)*100))
torch.save(net.state_dict(), './Models/lenet1.pt')

