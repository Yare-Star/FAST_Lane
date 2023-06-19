import torch,pdb
from model.CBAM import *
import torchvision
import torch.nn.modules

from typing import Callable, List, Optional
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial

class regnet(torch.nn.Module):
    def __init__(self, pretrained = True):
        super(regnet, self).__init__()
        # model = torchvision.models.regnet_y_400mf(pretrained = pretrained)
        # model = torchvision.models.regnet_y_800mf(pretrained=pretrained)
        # model = torchvision.models.regnet_y_8gf(pretrained=pretrained)
        model = torchvision.models.regnet_x_400mf(pretrained = pretrained)

        # self.fc = nn.Conv2d(440, 512, 1)  # 卷积核大小1X1，作用同全连接层 reg400Y
        self.fc = nn.Conv2d(400, 512, 1)    # 卷积核大小1X1，作用同全连接层 reg400X
        # self.fc = nn.Conv2d(784, 512, 1)  # 卷积核大小1X1，作用同全连接层 reg800
        # self.fc = nn.Conv2d(888, 512, 1)  # 卷积核大小1X1，作用同全连接层 1_6gf
        self.stem = model.stem
        self.trunk_output = model.trunk_output
        self.cbam = CBAMBlock(channel=512, reduction=16, kernel_size=9)

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk_output(x)
        x = self.fc(x)        
        x = self.cbam(x)
        return x

class resnet(torch.nn.Module):
    def __init__(self, layers, pretrained=True):
        super(resnet, self).__init__()
        if layers == '18':
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == '34':
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == '50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif layers == '101':
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif layers == '152':
            model = torchvision.models.resnet152(pretrained=pretrained)
        elif layers == '50next':
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif layers == '101next':
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        elif layers == '50wide':
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        elif layers == '101wide':
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        else:
            raise NotImplementedError
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.cbam512 = CBAMBlock(channel=512, reduction=16, kernel_size=9)
        self.cbam256 = CBAMBlock(channel=256, reduction=16, kernel_size=9)

    def forward(self, x):
        x = self.conv1(x)  # (64,144,400)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (64,72,200)
        x = self.layer1(x)  # (64,72,200)
        x2 = self.layer2(x)  # (128,36,100)
        x3 = self.layer3(x2)  # (256,18,50)
        x3 = self.cbam256(x3)
        x4 = self.layer4(x3)  # (512,9,25)
        x4 = self.cbam512(x4)
        return x2, x3, x4

