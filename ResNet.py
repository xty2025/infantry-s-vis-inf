import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #池化层来缩小特征图尺寸
        # Layer 1 (2 BasicBlocks with stride 1)
        self.layer1_1 = BasicBlock(64, 64, stride=1)
        self.layer1_2 = BasicBlock(64, 64, stride=1)
        
        # Layer 2 (2 BasicBlocks with stride 2 for downsampling)
        self.layer2_1 = BasicBlock(64, 128, stride=2)
        self.layer2_2 = BasicBlock(128, 128, stride=1)
        
        # Layer 3 (2 BasicBlocks with stride 2 for downsampling)
        self.layer3_1 = BasicBlock(128, 256, stride=2)
        self.layer3_2 = BasicBlock(256, 256, stride=1)
        
        # Layer 4 (2 BasicBlocks with stride 2 for downsampling)
        self.layer4_1 = BasicBlock(256, 512, stride=2)
        self.layer4_2 = BasicBlock(512, 512, stride=1)
        
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1_1(x)
        x = self.layer1_2(x)
        
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        
        x = self.layer4_1(x)
        x = self.layer4_2(x)

        x=F.adaptive_avg_pool2d(x,(1,1))#全局平均化
        x=torch.Flatten(x)#展平层
        x=self.fc(x)#全连接，输出分类结果
        return x
