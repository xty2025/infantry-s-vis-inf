import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import torch

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, planes, stride=1, downsample=None):
        super().__init__()
        self.bottleneck_convs = nn.Sequential(
            nn.Conv2d(in_channels, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion * planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bottleneck_convs(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class FPN(nn.Module):
    """
    一个集成了ResNet主干和FPN头的完整模型。
    初始化时需提供一个列表，定义ResNet每个阶段的Bottleneck数量。
    例如, ResNet-50 对应 [3, 4, 6, 3]。
    """
    def __init__(self, blocks_per_layer):
        super().__init__()
        self.in_channels = 64
        # Stem层
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.c2_stage = self._make_stage(64, blocks_per_layer[0])
        self.c3_stage = self._make_stage(128, blocks_per_layer[1], stride=2)
        self.c4_stage = self._make_stage(256, blocks_per_layer[2], stride=2)
        self.c5_stage = self._make_stage(512, blocks_per_layer[3], stride=2)

        fpn_out_channels = 256
        self.p5_lat_conv = nn.Conv2d(2048, fpn_out_channels, kernel_size=1)
        self.p4_lat_conv = nn.Conv2d(1024, fpn_out_channels, kernel_size=1)
        self.p3_lat_conv = nn.Conv2d(512, fpn_out_channels, kernel_size=1)
        self.p2_lat_conv = nn.Conv2d(256, fpn_out_channels, kernel_size=1)

        self.p4_smooth_conv = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.p3_smooth_conv = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.p2_smooth_conv = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)

    def _make_stage(self, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != BottleNeck.expansion * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, BottleNeck.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(BottleNeck.expansion * planes)
            )
        layers = []
        layers.append(BottleNeck(self.in_channels, planes, stride, downsample))
        self.in_channels = planes * BottleNeck.expansion
        for _ in range(1, num_blocks):
            layers.append(BottleNeck(self.in_channels, planes))
        return nn.Sequential(*layers)

    def _upsample_and_add(self, p, c_lat):
        _, _, H, W = c_lat.shape
        return F.interpolate(p, size=(H, W), mode='bilinear', align_corners=False) + c_lat

    def forward(self, x):
        c1_out = self.stem(x)
        c2_out = self.c2_stage(c1_out)
        c3_out = self.c3_stage(c2_out)
        c4_out = self.c4_stage(c3_out)
        c5_out = self.c5_stage(c4_out)

        p5_lat = self.p5_lat_conv(c5_out)
        p4_lat = self.p4_lat_conv(c4_out)
        p3_lat = self.p3_lat_conv(c3_out)
        p2_lat = self.p2_lat_conv(c2_out)

        p5_out = p5_lat
        p4_fused = self._upsample_and_add(p5_out, p4_lat)
        p3_fused = self._upsample_and_add(p4_fused, p3_lat)
        p2_fused = self._upsample_and_add(p3_fused, p2_lat)

        p4_out = self.p4_smooth_conv(p4_fused)
        p3_out = self.p3_smooth_conv(p3_fused)
        p2_out = self.p2_smooth_conv(p2_fused)

        return p2_out, p3_out, p4_out,
