import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from glob import glob
from torch.utils.data import Dataset, DataLoader
import os
import logging

#Read dataset and Net are the same as the test.

class ReadDataset(Dataset):
    def __init__(self, imgs_dir, data_label):
        self.imgs_dir = imgs_dir
        self.ids = data_label

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        imgs_file = self.imgs_dir+ str(idx) + '/' + str(i) + '.png'
        img = Image.open(imgs_file).convert('L')
        img = np.array(img)
        img = img.reshape(1,28,28)
        if img.max() > 1:
            img = img / 255
        return {'image': torch.from_numpy(img), 'label': torch.tensor(idx)}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.pool = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.linear1 = nn.Linear(16*4*4, 120)
        self.linear2 = nn.Linear(120,84)
        self.linear3 = nn.Linear(84,10)
    
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = x.view(-1, 10)
        return x

data_length = 10000
data_label = [-1] * data_length
prev_dir = './mnist_png/mnist_png/testing/'
after_dir = '.png'

for id in range(10):
    id_string = str(id)
    for filename in glob(prev_dir + id_string +'/*.png'):
        position = filename.replace(prev_dir+id_string+'/', '')
        position = position.replace(after_dir, '')
        data_label[int(position)] = id 

net = Net()
net = net.float()

all_data = ReadDataset(prev_dir, data_label)

batch_size = 4

test_loader = DataLoader(all_data, batch_size=batch_size, shuffle=True, num_workers=4)

print('finish loading data, ready to validate it')

PATH = './lenet5.pth'
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images = data['image']
        labels = data['label']

        outputs = net(images.float())

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))