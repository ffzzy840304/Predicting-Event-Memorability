from net import Net
import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
from torch import Tensor
from torchvision import models
import cv2
from PIL import Image
import torchvision
import csv
import numpy as np
import math

model=Net(19)
model.load_state_dict(torch.load('/path/to/models/best_mlp_model_nonormal_1.pt'))

rows=[]
target=[]
with open('split_name/R3_data_test_1.csv', 'r') as file:
    reader = csv.reader(file)
    i=0
    for row in reader:
      if i>=1:
        rows.append(row[0:-1])
        target.append(row[-1])       
      i=i+1

fsave=open('mlp_test_split1.txt','a+')
error=0
for i in range(0,len(rows)):
    row=rows[i][1:]
    row=np.array(row,dtype=float)
    row=Tensor(row)
    predict=model(row)
   # print(predict)
    #predict = predict.detach().numpy()
    predict=np.argmax(predict.detach().numpy())
    error=error+math.fabs((float(target[i])-predict))
    print(rows[i][0],predict,int(target[i]))
    fsave.writelines(rows[i][0]+' '+str(predict)+' '+target[i]+'\n')
fsave.close()
print(error/len(target))
