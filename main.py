import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import h5py
import Models
import csv
import codecs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.nn import Module, Parameter, init
from torch.nn import Conv2d, Linear, BatchNorm2d
from torch.nn.functional import relu
from torchvision import datasets, transforms


batch_size = 32
path=r'.\train.mat'
data=loadmat(path)               
data = data['horiz_signals'][:] #Real 为打开mat的实际文件名
train_data = data[:,:-1].astype(np.float32)
train_label = data[:,-1].astype(np.float32)


class Train_data(Dataset):
    def __init__(self,x,y):
        # self.x = x.unsqueeze(dim=1)
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")

train_loader = torch.utils.data.DataLoader(Train_data(train_data, train_label), batch_size= batch_size, shuffle=True)
#test_loader = torch.utils.data.DataLoader(train_set_imag, batch_size= batch_size, shuffle=True)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = getattr(Models, 'TFN_STTF') \
            (in_channels=1, out_channels=1, kernel_size=8,
             clamp_flag=True, mid_channel=70).to(device)
print(model)
# print("=== Initialized Weight and Superparameters ===")       
# weight_init, superparams_init = model.getweight() 
# weight_init = np.asarray(weight_init).reshape(20,19)
# superparams_init = np.asarray(superparams_init)
# data_write_csv(".\\weight_init.csv", weight_init)
# data_write_csv(".\\superparams_init.csv", superparams_init)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
try:
    model=torch.load(r'C:\Users\Yu\Desktop\Cn\Complexnet\weight\model')
except:
# train steps
    train_loss = []
    for epoch in range(60):
        for batch_idx, (x, label) in enumerate(train_loader):
            
            x = x.unsqueeze(1)
            x , label = x.to(device), label.to(device)
        
            optimizer.zero_grad()

            output = model(x).squeeze(1)
            loss = F.mse_loss(output, label)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            # if batch_idx % 10 == 0:
            #     print('Train Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch,
            #     batch_idx * len(x_real), 
            #     len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), 
            #     loss.item())
            # )
        if epoch % 5 == 0:
            print("epoch is {}, loss is {:.4f}".format(epoch, loss.item()))
    # torch.save(model, r'C:\Users\caoyu\Desktop\可解释性\TFN-main\weight\model1')
    plt.plot(train_loss)

# print("=== Initialized Weight and Superparameters ===")
# weight, superparams = model.getweight()
# weight = np.asarray(weight).reshape(20,19)
# superparams = np.asarray(superparams)
# data_write_csv(".\\weight.csv", weight)
# data_write_csv(".\\superparams.csv", superparams)

path=r'.\test2-6.mat'
data=loadmat(path)               
data = data['horiz_signals'][:] #Real 为打开mat的实际文件名
test_data = data[:,:-1].astype(np.float32)
test_label = data[:,-1].astype(np.float32)
test_loader = torch.utils.data.DataLoader(Train_data(test_data,test_label), batch_size= 8, shuffle=False)
predict=[]
for batch_idx, (test_x,test_label) in enumerate(test_loader):
    with torch.no_grad():
        test_data = test_x.unsqueeze(1)
        test_data = test_data.to(device)
        output = model(test_data)
        predict.append(output)
predict__out=[]
for a in predict:
    for b in a:
        predict__out.append(b)
#model = torch.load(r'C:\Users\Yu\Desktop\Cn\Complexnet\weight\model1')


predict__out = torch.tensor(predict__out, device = 'cpu')
predict__out=np.asarray(predict__out)
predict__out=predict__out.reshape(len(predict__out),1)

data_write_csv(".\\parametres.csv", predict__out)
#plt.plot(output)
