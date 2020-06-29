import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .modules import *

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, with_bn=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.relu = nn.ReLU()
        self.with_bn = with_bn
        if with_bn:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = lambda x: x
        self.reset_parameters()

    def forward(self, x):
        return self.norm(self.relu(self.conv(x)))

    def reset_parameters(self):
        nn.init.xavier_uniform(self.conv.weight)
        self.conv.bias.data.fill_(0.2)
        if self.with_bn:
            self.norm.reset_parameters()


class SID(nn.Module):
    def __init__(self, with_bn=False, threshold=3):
        super(SID, self).__init__()
        self.with_bn = with_bn
        self.preprocessing = SRMConv2d(1, 0)
        self.TLU = nn.Hardtanh(-threshold, threshold, True)
        if with_bn:
            self.norm1 = nn.BatchNorm2d(30)
        else:
            self.norm1 = lambda x: x
        self.block2 = ConvBlock(30, 30, 3, with_bn=self.with_bn)
        self.block3 = ConvBlock(30, 30, 3, with_bn=self.with_bn)
        self.block4 = ConvBlock(30, 30, 3, with_bn=self.with_bn)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.block5 = ConvBlock(30, 32, 5, with_bn=self.with_bn)
        self.pool2 = nn.AvgPool2d(3, 2)
        self.block6 = ConvBlock(32, 32, 5, with_bn=self.with_bn)
        self.pool3 = nn.AvgPool2d(3, 2)
        self.block7 = ConvBlock(32, 32, 5, with_bn=self.with_bn)
        self.pool4 = nn.AvgPool2d(3, 2)
        self.block8 = ConvBlock(32, 16, 3, with_bn=self.with_bn)
        self.block9 = ConvBlock(16, 16, 3, with_bn=self.with_bn)
        self.ip1 = nn.Linear(4 * 16, 256)
        self.ip2 = nn.Linear(256, 2)
        self.reset_parameters()

    def forward(self, x):
        x = x.float()
        x = self.preprocessing(x)
        x = self.TLU(x)
        x = self.norm1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool1(x)
        x = self.block5(x)
        x = self.pool2(x)
        x = self.block6(x)
        x = self.pool3(x)
        x = self.block7(x)
        x = self.pool4(x)
        x = self.block8(x)
        x = self.block9(x)
        # print(x.size())
        bs = x.shape[0]
        x_max = torch.max(x.reshape(bs, 16, -1), dim=2)
        x_max = x_max[0]
        x_min = torch.min(x.reshape(bs, 16, -1), dim=2)
        x_min = x_min[0]
        x_variance = torch.var(x.reshape(bs, 16, -1), dim=2)
        x_mean = torch.mean(x.reshape(bs, 16, -1), dim=2)
        x_stack = torch.stack([x_max, x_min, x_variance, x_mean], dim=2)
        x = x_stack.view(x_stack.size(0), -1)
        x = self.ip1(x)
        x = self.ip2(x)
        return x

    def reset_parameters(self):
        for mod in self.modules():
            if isinstance(mod, SRMConv2d) or isinstance(mod, nn.BatchNorm2d) or \
                    isinstance(mod, ConvBlock):
                mod.reset_parameters()
            elif isinstance(mod, nn.Linear):
                nn.init.normal(mod.weight, 0., 0.01)
                mod.bias.data.zero_()
