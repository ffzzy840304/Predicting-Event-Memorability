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

    def __init__(self):
        super(Net, self).__init__()
        
        self.model=models.vgg16(pretrained=True)
        self.model.classifier = self.model.classifier[:-1]
        
        self.fc1=nn.Linear(4096,256)
        self.drop1=nn.Dropout(p = 0.5)

        self.fc2=nn.Linear(256,64)
        self.drop2=nn.Dropout(p = 0.5)

        self.fc_out=nn.Linear(64,10)
        """
        self.model=models.resnet152(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, 1)
        """
    def forward(self, x):
        x=self.model(x) 
        
        x=self.drop1(F.relu(self.fc1(x)))
        x=self.drop2(F.relu(self.fc2(x)))

        out = self.fc_out(x)
        
        return out
