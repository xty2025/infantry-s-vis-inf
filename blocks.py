import torch
import torchvision
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import os
#通道数压缩比例，用于在全连接层中减小特征维度。默认为 16，即在激励（Excitation）操作中将通道数压缩到原始通道数的 1/16。
#SE块的作用是通过自适应地调整通道间的权重，使得网络能够更好地关注重要的特征通道。
class SEBlock(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.fc1=nn.Linear(channel,channel//reduction)
        self.fc2=nn.Linear(channel//reduction,channel)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        y=x.mean(dim=0,keepdim=True)if x.dim==1 else x
        y=self.fc1(y)
        y=self.relu(y)
        y=self.fc2(y)
        y=self.sigmoid(y)
        return x*y
    
class ECABlock(nn.Module):
    def __init__(self,in_channels,k_size=3):
        super(ECABlock,self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=k_size,padding=1,bias=False)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        b,c,_,_=x.size()
        y=self.avg_pool(x).view(b,c,1)
        y=self.conv(y.transpose(1,2)).transpose(1,2)
        y=self.sigmoid(y).view(b,c,1,1)
        return x*y.expand_as(x)
    