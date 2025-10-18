import torch
import torch.nn as nn
#通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // reduction, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out.expand_as(x)
#空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x * x
#注意力融合，同时用到空间和通道，而且采用maxpool和avgpool的形式。
class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
    

#RGB+Depth的应用中加：
import torch
import torch.nn as nn
import torchvision.models as models

# 假设你已经定义了 CBAM 模块（ChannelAttention 和 SpatialAttention）

class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# 假设 ChannelAttention 和 SpatialAttention 已经定义好

class FusionModelWithCBAM(nn.Module):
    def __init__(self, in_channels_rgb=3, in_channels_depth=1, cbam_reduction=16, kernel_size=7):
        super(FusionModelWithCBAM, self).__init__()
        
        # 定义 CBAM 模块
        self.cbam = CBAM(in_channels_rgb + in_channels_depth, reduction=cbam_reduction, kernel_size=kernel_size)
        
        # 你可以在这里加入自己的网络层，例如卷积层，池化层等
        self.conv1 = nn.Conv2d(in_channels_rgb + in_channels_depth, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 112 * 112, 10)  # 假设输出是10类的分类任务

    def forward(self, rgb_img, depth_img):
        # 拼接RGB图像和深度图像
        x = torch.cat((rgb_img, depth_img), dim=1)  # 拼接在通道维度
        
        # 通过CBAM模块进行特征增强
        x = self.cbam(x)
        
        # 接下来是卷积操作，特征提取等
        x = self.conv1(x)
        x = self.pool(x)
        
        # 假设我们进行分类任务，展平并通过全连接层
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        
        return x

# 假设你已经准备好了输入的RGB图像和深度图像
rgb_img = torch.randn(8, 3, 224, 224)  # 8张RGB图像，尺寸为224x224
depth_img = torch.randn(8, 1, 224, 224)  # 8张深度图像，尺寸为224x224

# 初始化模型
model = FusionModelWithCBAM(in_channels_rgb=3, in_channels_depth=1, cbam_reduction=16, kernel_size=7)

# 前向传播
output = model(rgb_img, depth_img)
print(output.shape)  # 输出的形状（8, 10），表示8个样本，10个类别
