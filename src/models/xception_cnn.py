"""
Modified Xception CNN backbone for exoplanet detection.
Leverages spatial structure for robust feature extraction from phase-folded images.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)

class XceptionCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.block1 = XceptionBlock(in_channels, 32)
        self.block2 = XceptionBlock(32, 64)
        self.block3 = XceptionBlock(64, 128)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
