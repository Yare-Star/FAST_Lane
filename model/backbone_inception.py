import torch, pdb
from model.CBAM import *
import torchvision
import torch.nn.modules

from typing import Callable, List, Optional
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial


class regnet(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(regnet, self).__init__()
        # model = torchvision.models.regnet_y_400mf(pretrained = pretrained)
        # model = torchvision.models.regnet_y_800mf(pretrained=pretrained)
        # model = torchvision.models.regnet_y_8gf(pretrained=pretrained)
        model = torchvision.models.regnet_x_400mf(pretrained=pretrained)

        # self.fc = nn.Conv2d(440, 512, 1)  # 卷积核大小1X1，作用同全连接层 reg400Y
        self.fc = nn.Conv2d(400, 512, 1)  # 卷积核大小1X1，作用同全连接层 reg400X
        # self.fc = nn.Conv2d(784, 512, 1)  # 卷积核大小1X1，作用同全连接层 reg800
        # self.fc = nn.Conv2d(888, 512, 1)  # 卷积核大小1X1，作用同全连接层 1_6gf
        self.stem = model.stem
        self.trunk_output = model.trunk_output
        self.cbam = CBAMBlock(channel=512, reduction=16, kernel_size=9).cuda()

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

        # model.layer1[0].conv1 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='same', dilation=1, bias=False)  # 增加空洞卷积

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.cbam512 = CBAMBlock(channel=512, reduction=16, kernel_size=9)  # res18||res34
        # self.cbam512_res50 = CBAMBlock(channel=2048, reduction=16, kernel_size=9)  # res50
        # self.cbam256 = CBAMBlock(channel=256, reduction=16, kernel_size=9)

        ########################changed3
        # self.layer5 = torch.nn.Sequential(
        #     torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), bias=False),
        #     torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding='same', bias=False),
        #     torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding='same', bias=False),
        #     torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding='same', bias=False),
        #     torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     torch.nn.ReLU(inplace=True),
        # )

    def forward(self, x):
        x = self.conv1(x)  # (64,144,400)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (64,72,200)
        x = self.layer1(x)  # (64,72,200)
        x2 = self.layer2(x)  # (128,36,100)
        x3 = self.layer3(x2)  # (256,18,50)
        # x3 = self.cbam256(x3)
        x4 = self.layer4(x3)  # (512,9,25)
        x4 = self.cbam512(x4)  # res18||res34
        # x4 = self.cbam512_res50(x4)  # res50
        #####changed3
        # x3 = self.layer5(x3)
        return x2, x3, x4


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)  # 后面接BN时，不需要bias
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block35(nn.Module):

    def __init__(self, scale=0.2):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)  # res18||res34
        # self.branch0 = BasicConv2d(1024, 32, kernel_size=1, stride=1)  # res50

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),  # res18||res34
            # BasicConv2d(1024, 32, kernel_size=1, stride=1),  # res50
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),  # res18||res34
            # BasicConv2d(1024, 32, kernel_size=1, stride=1),  # res50
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(128, 256, kernel_size=1, stride=1)  # res18||res34
        # self.conv2d = nn.Conv2d(128, 1024, kernel_size=1, stride=1)  # res50
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class inception(torch.nn.Module):
    def __init__(self):
        super(inception, self).__init__()
        self.net = nn.Sequential(
            Block35(scale=0.2),
            Block35(scale=0.2),
            Block35(scale=0.2),
        )
        self.down = nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), bias=False),  # res18||res34
            torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # torch.nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), bias=False),  # res50
            # torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.down(x)
        return x
