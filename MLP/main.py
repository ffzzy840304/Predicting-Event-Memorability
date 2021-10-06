import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from net import Net

val_loss_min = np.Inf

df = pd.read_csv("split_name/R3_data_train_1.csv")
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]
df_val=pd.read_csv("split_name/R3_data_test_1.csv")
X_v=df_val.iloc[:, 1:-1]
y_v=df_val.iloc[:, -1]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, stratify=y, random_state=69)
#X_val,X_vt,y_val,y_vt=train_test_split(X_v, y_v, test_size=10, stratify=y_v, random_state=21)

scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X)
X_train, y_train = np.array(X), np.array(y)

y_train = y_train.astype(float)

#X_val = scaler.fit_transform(X_v)
X_val, y_val = np.array(X_v), np.array(y_v)

y_val = y_val.astype(float)
class R3(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


def ordinal_loss(y_pred, y_true): 
    """   
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)   
    """   
    diff=(torch.abs(torch.argmax(y_true)-torch.argmax(y_pred,dim=1))).float()/(y_pred.size()[0]-1)
    #print((torch.argmax(y_true)))    
    weights=diff
    #print(weights)
    return torch.mean((1.0 + weights) * nn.CrossEntropyLoss()(y_pred,y_true))


train_dataset = R3(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
val_dataset = R3(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 0.001
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
print(use_gpu)
#torch.cuda.set_device(1)
model = Net(27) # need to change to the number of features for different configurations.
print(model)
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_stats = {
    'train': [],
    "val": []
}

print("Begin training.")
for e in range(1, EPOCHS+1):
    
    # TRAINING
    train_epoch_loss = 0
    model.train()
    
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        
        y_train_pred = model(X_train_batch)
        
        train_loss = ordinal_loss(y_train_pred, y_train_batch.long())    #criterion(y_train_pred, y_train_batch.unsqueeze(1))
        
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()

    # VALIDATION    
    with torch.no_grad():
        
        val_epoch_loss = 0
        
        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = model(X_val_batch)
                        
            val_loss = ordinal_loss(y_train_pred, y_train_batch.long())  #criterion(y_val_pred, y_val_batch.unsqueeze(1))
            
            val_epoch_loss += val_loss.item()

    train_epoch_loss=train_epoch_loss/len(train_loader)
    val_epoch_loss=val_epoch_loss/len(val_loader)  
    if val_epoch_loss<val_loss_min:                          
       val_loss_min=val_epoch_loss
       torch.save(model.state_dict(), 'mlp_models/best_mlp_model_nonormal_1.pt')
    print(e)
    print(train_epoch_loss, val_epoch_loss)
