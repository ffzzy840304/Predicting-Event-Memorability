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

    def __init__(self,num_features=28):
        super(Net, self).__init__()
        self.model=models.vgg16(pretrained=True)
        self.model.classifier = self.model.classifier[:-1]
        
        self.mlp=nn.Linear(num_features,1024)
        self.fc2=nn.Linear(5120,256)
        self.fc3=nn.Linear(256,64)
        self.fc_out=nn.Linear(64,10)
        self.drop_2=nn.Dropout(p = 0.12)
        self.drop_3=nn.Dropout(p = 0.39)
    def forward(self, x, fea):
        x=self.model(x) 
    
        fea1=self.mlp(fea)
        out = torch.cat((fea1, x), dim=1)
        out = self.drop_2(F.relu(self.fc2(out)))
        out = self.drop_3(F.relu(self.fc3(out)))
        out = self.fc_out(out)
        return out
