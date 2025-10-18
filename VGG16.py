import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义 VGG Block
#也可不通过block实现。直接写conv,conv,maxpool
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super(VGGBlock, self).__init__()
        
        # 定义多个卷积层
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))  # 3x3 卷积
            layers.append(nn.ReLU(inplace=True))  # ReLU 激活函数
            in_channels = out_channels  # 每个卷积层的输出作为下一个卷积层的输入
        
        self.conv_block = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 最大池化，步长为2

    def forward(self, x):
        x = self.conv_block(x)  # 通过卷积层
        x = self.pool(x)  # 通过池化层
        return x

# 定义 VGG-16 网络
class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        
        # 构建 VGG 的 5 个块，每个块由多个卷积层和一个池化层组成
        self.block1 = VGGBlock(3, 64, num_convs=2)  # Block1: 输入3通道，输出64通道，2个卷积层
        self.block2 = VGGBlock(64, 128, num_convs=2)  # Block2: 输入64通道，输出128通道，2个卷积层
        self.block3 = VGGBlock(128, 256, num_convs=3)  # Block3: 输入128通道，输出256通道，3个卷积层
        self.block4 = VGGBlock(256, 512, num_convs=3)  # Block4: 输入256通道，输出512通道，3个卷积层
        self.block5 = VGGBlock(512, 512, num_convs=3)  # Block5: 输入512通道，输出512通道，3个卷积层
        
        # 全连接层部分
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)  # 输入尺寸为 512 * 7 * 7（全局池化后）
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)  # 输出类别数
        
    def forward(self, x):
        x = self.block1(x)  # 通过 Block1
        x = self.block2(x)  # 通过 Block2
        x = self.block3(x)  # 通过 Block3
        x = self.block4(x)  # 通过 Block4
        x = self.block5(x)  # 通过 Block5
        
        x = x.view(x.size(0), -1)  # 展平为一维向量 (batch_size, 512*7*7)
        x = F.relu(self.fc1(x))  # 通过全连接层1
        x = F.relu(self.fc2(x))  # 通过全连接层2
        x = self.fc3(x)  # 最后一层是输出层
        return x

# 实例化模型
model = VGG16(num_classes=1000)  # 输出 1000 类，适用于 ImageNet

# 打印模型结构
print(model)
