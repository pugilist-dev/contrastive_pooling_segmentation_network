#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 12:12:53 2021

@author: raj
"""

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
from torchvision import models
from model.DRN import drn_d_54

device = 'cuda'

class PyramidPooling(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_ch / 4)
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=inter_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=in_ch, out_channels=inter_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=in_ch, out_channels=inter_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=in_ch, out_channels=inter_channels, kernel_size=1)
        self.out = nn.Conv2d(in_channels = in_ch * 2, out_channels=out_ch, kernel_size=1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x

class my_drn(nn.Module):
    def __init__(self):
        super(my_drn, self).__init__()
        drn = drn_d_54()
        drn = list(drn.children())
        self.conv = nn.Sequential(*drn[:-2])
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv2d(512, 19, 1),
        )
        self.ppm = PyramidPooling(512, 512)
    def forward(self, images):
        size = images.size()[2:]
        feat_map = self.conv(images)
        ppm_out = self.ppm(feat_map)
        out = feat_map + ppm_out
        x = F.interpolate(out, size, mode='bilinear', align_corners=True)
        x = self.classifier(x)
        outputs = []
        outputs.append(x)
        return tuple(outputs)
    
def get_drn():
    model = my_drn()
    return model
        