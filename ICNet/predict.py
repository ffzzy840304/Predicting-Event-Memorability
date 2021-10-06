import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import cv2
from net_cnn import Net
import numpy as np

from Dataset import  *

BATCH_SIZE=64
LEARNING_RATE = 0.001
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.set_device(1)
use_gpu = torch.cuda.is_available()
print(use_gpu)
dataset_root='datasets/lamem/'
image_root='datasets/lamem/images/'
split_root='datasets/lamem/splits/'

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


def load_dataset(split='train_1', train=True, demo=False,batch_size=512, dataset_root='datasets/lamem/',
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

def predict(model, test_loader,memory_save_dir):

    model.eval()

    predictions = []
    targets = []
    output = None
    outputs = None
    alphas = None
    img_names = []

    batches = 0
    img_inference_took = 0
    for data, features,target, names in test_loader:
        img_names += names
     
        target = target.float()
       
        data, features, target = data.to(device), features.to(device), target.to(device)

        data, features, target = Variable(data, volatile=True),Variable(features), Variable(target)

        # print(batches, len(predictions))
        #batch_inference_start = time.time()
        output = model(data)
        """
        batch_inference_took = time.time() - batch_inference_start
        img_inference_took += batch_inference_took / data.size(0)
        batches += 1
        """
        
        #output= output.cpu().data.numpy().item()
        #output=output.cpu().data.item()
        fsave=open(memory_save_dir,'a+')
        for m in range(0,len(img_names)):
            fsave.writelines(img_names[m]+' '+str(np.argmax(output[m].cpu().data.numpy()))+' '+'0.0'+'\n')
        fsave.close() 
    return img_names[0], output[0].cpu().data.numpy()


def predict_memorability(model, images_path,memory_save_dir):
    dataset = ICset(split=images_path, demo=True,
                         transform=test_transform)
    num_workers = 1
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
                                                  num_workers=num_workers)

    results=predict(model,loader,memory_save_dir)
    return results

model=Net()
model.load_state_dict(torch.load('fusion_models/best_fusion_model_split_5.pt'))
model.to(device)

image_dir='photo_subs_split5'
memory_save_dir='memory_icnet_split5.txt'
f_list=[]
my_dirs=[d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir,d))]
for m in range(0,len(my_dirs)):
    f_list.append(my_dirs[m])
for m in range(0,len(f_list)):
    sub_dir=image_dir+'/'+f_list[m]
    result=predict_memorability(model,sub_dir,memory_save_dir)
       

