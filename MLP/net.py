import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
from torchvision import models
import cv2
from PIL import Image
import torchvision

class Net(nn.Module):

    def __init__(self, num_features):
        super(Net, self).__init__()
        self.fc1=nn.Linear( num_features, 16)
        self.relu= nn.ReLU()
        self.fc2=nn.Linear(16,32)
        self.fc3=nn.Linear(32,16)
        self.fc_out=nn.Linear(16,10)
       
    def forward(self, x):
        print(x.shape)
        x = self.relu(self.fc1(x))     
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc_out(x)
       # print(x)
        return x