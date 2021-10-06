import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import cv2
from net_cnn import Net
import numpy as np
import torch.nn.functional as F
from Dataset import  *

TRAIN_SPLIT='train_5'
VAL_SPLIT='val_5'
SAVE_SPLIT='split_5'

BATCH_SIZE=32
LEARNING_RATE = 0.0001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
print(use_gpu)
dataset_root='datasets/r3/'
image_root='datasets/r3/images/'
split_root='datasets/r3/splits/'
model=Net()
model.to(device)

train_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Test
test_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def load_dataset(split='train_1', train=True, demo=False,batch_size=512, dataset_root='datasets/r3/',
                    drop_last=True, num_workers=2):

    dataset = ICset(image_root,
                     split_root,
                     split=split,
                     demo=demo,
                     transform=train_transform if train else test_transform)

    if len(dataset.data) < batch_size:
        batch_size = len(dataset.data)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last,
                                              num_workers=num_workers)

    print("Loaded dataset:")
    print("\t", dataset.split_file)
    print("\ttrain: ", train)
    print("\tsamples: ", len(dataset.data))
    print("\tbatch size: ", batch_size)
    print("\tnum_workers: ", num_workers)
    return data_loader

def save_checkpoint(filename,model):
    dirs, _ = os.path.split(filename)
    os.makedirs(dirs, exist_ok=True)
    print('Saving checkpoint: ', filename)
    torch.save({'model': model.state_dict()}, filename)


def load_checkpoint(filename):

    if filename.strip() == '':
        return False
    try:
        print('Loading checkpoint: ', filename)
        cpnt = torch.load(filename, map_location=lambda storage, loc: storage)
        self.experiment_path, filename = os.path.split(filename)
    except FileNotFoundError:
        print("Cannot open file: ", filename)
        self.model_weights_current = ''
        return False
    try:
        self.model.load_weights(cpnt['model'])
    except:
        self.model.load_state_dict(cpnt['model'])

    return True


criterion =nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


train_loss_min= np.Inf

def ordinal_loss(y_pred, y_true): 
    """   
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)   
    """   
    diff=(torch.abs(torch.argmax(y_true)-torch.argmax(y_pred,dim=1))).float()/(y_pred.size()[0]-1)
    print(torch.argmax(y_true),torch.argmax(y_pred,dim=1) )
    weights=diff
    #print(weights)
    return torch.mean((1.0 + weights) * nn.CrossEntropyLoss()(y_pred,y_true))

train_data_loader = load_dataset(split=TRAIN_SPLIT, train=True, demo=False, batch_size=BATCH_SIZE,
                                         dataset_root=dataset_root)

test_data_loader = load_dataset(split=VAL_SPLIT, train=False, demo=False, batch_size=BATCH_SIZE,
                                        dataset_root=dataset_root)
print(len(train_data_loader),len(test_data_loader))

for epoch in range(0,50):
    print("Epoch {0} is training.".format(epoch))
    #train_epoch(model, epoch,train_data_loader,test_data_loader)
    total_train_loss=0
    total_val_loss=0
    start_time = batch_time = time.time()
    for batch_idx, (data, fea, target, _) in enumerate(train_data_loader):
        #if params.use_cuda:
        data, fea,target = data.to(device), fea.to(device), target.float().to(device)
           # print(data.shape)
        data, fea, target = Variable(data), Variable(fea), Variable(target)
        optimizer.zero_grad()
        output=model(data)
 
        #print(output,(target.long()))
        train_loss = ordinal_loss(output,target.long())  #criterion(output, target.unsqueeze(1))
        #print(train_loss)
        train_loss.backward()
        optimizer.step()
        total_train_loss=total_train_loss+train_loss.item()
    
    with torch.no_grad():              
        model.eval()
        for batch_idx, (data, fea, target, _) in enumerate(test_data_loader):
           #if params.use_cuda:
           data, fea,target = data.to(device), fea.to(device), target.long().to(device)
           # print(data.shape)
           data, fea, target = Variable(data), Variable(fea), Variable(target)

           output=model(data)
           #print(output,target)
           val_loss = ordinal_loss(output,target.long())  #criterion(output, target.unsqueeze(1))
           total_val_loss=total_val_loss+val_loss.item()
    
    total_train_loss=total_train_loss/len(train_data_loader)
    total_val_loss=total_val_loss/len(test_data_loader)
    
    print(total_train_loss,total_val_loss)
    if total_train_loss<train_loss_min:
       train_loss_min=total_train_loss
       print("better model trained!")
       torch.save(model.state_dict(), '/path/to/save/model/best_model_'+SAVE_SPLIT+'.pt')





