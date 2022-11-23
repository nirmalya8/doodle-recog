import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
import torchvision.transforms as transforms
from utils import load_dataset,to_categorical,horizontal_flip, vertical_flip
from dataset import DoodleDataset
from model import LeNet 


image_size = 28
x_train,x_test,y_train,y_test,classes = load_dataset('./Data')
x_train = x_train.reshape(x_train.shape[0], image_size, image_size, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], image_size, image_size, 1).astype('float32')

test_dataset = DoodleDataset(x_test, y_test, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
print(y_train.shape,y_test.shape)
print(x_train.shape,x_test.shape)

model = LeNet()
model.load_state_dict(torch.load("./Models/lenet1.pt"))
model.eval()
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # data, target = data.to('cuda'), target.to('cuda')
            output = model(data)
            #test_loss += f.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
            # print(pred, target)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
         correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


test(model, test_loader)