import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义经典CNN模型
class ClassicCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ClassicCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = nn.Linear(256 * 28 * 28, 512)  # 256通道，28x28的特征图
        self.fc2 = nn.Linear(512, num_classes)    # 输出类别数为num_classes
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))        
        x = self.pool2(F.relu(self.conv2(x)))      
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 28 * 28)  # 展平为(batch_size, 256 * 28 * 28)        
        # 全连接层1
        x = F.relu(self.fc1(x))        
        # 全连接层2
        x = self.fc2(x)        
        return x