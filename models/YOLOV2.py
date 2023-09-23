import torch
import torch.nn as nn
from .ReferenceNet import ReferenceNet


class YOLOV2(nn.Module):
    def __init__(self, B=5, C=20):
        super(YOLOV2, self).__init__()
        in_channels = 3
        out_channels = 1024
        add_channels = 1024
        region_channels = (5 + C) * B
        self.features = ReferenceNet(in_channels=in_channels, out_channels=out_channels)
        self.additional = nn.Sequential(
            nn.Conv2d(out_channels, add_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(add_channels),
            nn.LeakyReLU(0.1)
        )
        self.region_conv = nn.Conv2d(add_channels, region_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.features(x)
        x = self.additional(x)
        x = self.region_conv(x)
        return x