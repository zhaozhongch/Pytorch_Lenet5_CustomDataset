import torch
import torchvision
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from glob import glob
from torch.utils.data import Dataset, DataLoader
import os

all_image_list = []
class_id = []

#ReadDataset is used to parse the custom dataset
class ReadDataset(Dataset):
    def __init__(self, imgs_dir, data_label):
        self.imgs_dir = imgs_dir
        self.ids = data_label

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]

        #find image `i.png` in folder `training/idx`. Convert to grayscale
        imgs_file = self.imgs_dir+ str(idx) + '/' + str(i) + '.png'
        img = Image.open(imgs_file).convert('L')

        #convert image to float numpy array
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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

data_length = 60000
data_label = [-1] * data_length
prev_dir = './mnist_png/mnist_png/training/'
after_dir = '.png'

#For every file in the dataset folder, combine the id with its sequence.
#For example, in `training/0/21.png`, the image has label 0, `then data_label[21]` will be 0 
for id in range(10):
    id_string = str(id)
    for filename in glob(prev_dir + id_string +'/*.png'):
        position = filename.replace(prev_dir+id_string+'/', '')
        position = position.replace(after_dir, '')
        data_label[int(position)] = id 

#Define net and convert it to float
net = Net()
net = net.float()

#define optimization criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

dir_imgs = 'mnist_png/mnist_png/training/'
all_data = ReadDataset(dir_imgs, data_label)

batch_size = 8

train_loader = DataLoader(all_data, batch_size=batch_size, shuffle=True, num_workers=8)

print('Load data finish, ready to train')

for epoch in range(1):
    net.train()

    epoch_loss = 0.0
    batch_num = 0
    for training_batch in train_loader:
        batch_num = batch_num + 1

        images = training_batch['image']
        labels = training_batch['label']

        optimizer.zero_grad()

        outputs = net(images.float( ))

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # print every 2000 mini-batches
        if batch_num % 200 == 199:    
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_num + 1, epoch_loss / 2000))
            epoch_loss = 0.0

print('Finish training')

PATH = './lenet5.pth'
torch.save(net.state_dict(), PATH)

    
