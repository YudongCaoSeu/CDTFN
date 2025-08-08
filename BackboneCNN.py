#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import torch
from utils.mysummary import summary

# ----------------------------inputsize = 1024-------------------------------------------------------------------------
class ComplexConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self,input_r, input_i):
        assert(input_r.size() == input_i.size())
        return self.conv_r(input_r)-self.conv_i(input_i), self.conv_r(input_i)+self.conv_i(input_r)

class ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self,input_r, input_i):
        return self.fc_r(input_r)-self.fc_i(input_i), self.fc_r(input_i)+self.fc_i(input_r)

def complex_relu(input_r, input_i):
    relu = nn.ReLU()
    return relu(input_r), relu(input_i)

class ComplexSequential(nn.Module):
    def __init__(self, *layers):
        super(ComplexSequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, input_r, input_i):
        for layer in self.layers:
            input_r, input_i = layer(input_r, input_i)
        return input_r, input_i
    def __getitem__(self, idx):
        return self.layers[idx]
    def replace_layer(self, idx, new_layer):
        # 确保新层是 nn.Module 的实例
        assert isinstance(new_layer, nn.Module), "new_layer must be an instance of nn.Module"
        self.layers[idx] = new_layer

class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=10,kernel_size=15,**kwargs):
        super(CNN, self).__init__()

        self.layer1 = ComplexSequential(
            ComplexConv1d(in_channels, 20, kernel_size=kernel_size, bias=True),
        )

        self.layer2 = ComplexSequential(
            ComplexConv1d(20, 50, kernel_size=3, bias=True),
        )

        # self.layer4 = nn.Sequential(
        #     nn.Conv1d(64, 128, kernel_size=3, bias=True),  # 128,500
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveMaxPool1d(4))  # 128, 4

        self.fc1 = ComplexLinear(127550, 500)
        self.fc2 = ComplexLinear(500, 100)
        self.fc3 = ComplexLinear(100, 50)
        self.fc4 = ComplexLinear(50, 10)
        self.fc5 = nn.Linear(10, out_channels)

    def forward(self, x1,x2):
        if len(x1.shape) == 4:
            x1 = torch.squeeze(x1)
        if len(x2.shape) == 4:
            x2 = torch.squeeze(x2)
        x_r,x_i = self.layer1(x1,x2)
        x_r,x_i = complex_relu(x_r,x_i)
        x_r,x_i = self.layer2(x_r,x_i)
        x_r,x_i = complex_relu(x_r,x_i)

        x_r = x_r.view(x_r.size(0), -1)
        x_i = x_i.view(x_i.size(0), -1)

        x_r,x_i = self.fc1(x_r,x_i)
        x_r,x_i = complex_relu(x_r,x_i)
        x_r,x_i = self.fc2(x_r,x_i)
        x_r,x_i = self.fc3(x_r,x_i)
        x_r,x_i = self.fc4(x_r,x_i)
        # x = self.layer5(x)
        x = torch.sqrt(torch.pow(x_r,2)+torch.pow(x_i,2))
        x = self.fc5(x)
        x = torch.sigmoid(x)
        return x
    
    def feature(self, x1,x2):
        if len(x1.shape) == 4:
            x1 = torch.squeeze(x1)
        if len(x2.shape) == 4:
            x2 = torch.squeeze(x2)
        x_r,x_i = self.layer1(x1,x2)
        x_r,x_i = complex_relu(x_r,x_i)
        x_r,x_i = self.layer2(x_r,x_i)
        x_r,x_i = complex_relu(x_r,x_i)

        x_r = x_r.view(x_r.size(0), -1)
        x_i = x_i.view(x_i.size(0), -1)

        x_r,x_i = self.fc1(x_r,x_i)
        x_r,x_i = complex_relu(x_r,x_i)
        x_r,x_i = self.fc2(x_r,x_i)
        x_r,x_i = self.fc3(x_r,x_i)
        x_r,x_i = self.fc4(x_r,x_i)
        return x_r, x_i 


if __name__ == '__main__':
    model = CNN()
    info = summary(model, (1, 1024), batch_size=-1, device="cpu")
    print(info)