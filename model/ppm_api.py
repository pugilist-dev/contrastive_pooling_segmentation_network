#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 17:12:24 2021

@author: rajiv
"""
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
from torchvision import models
from model.DRN import drn_d_22

device = 'cuda'

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


"""
class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.LeakyReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(True)
        )

    def forward(self, x):
        return self.conv(x)
    
class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 9)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 15)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 24)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 39)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x
"""

class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()
        drn = drn_d_22()
        drn = list(drn.children())
        self.conv = nn.Sequential(*drn[:-2])
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True)
        )
        #self.conv1 = _ConvBNReLU(in_channels=1024, out_channels= 1792, kernel_size=3,
        #                           stride=1, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1792),
            nn.LeakyReLU(True)
        )
        #self.conv2 = _ConvBNReLU(in_channels=1792, out_channels= 1024, kernel_size=3,
        #                           stride=1, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(True)
        )
        #self.conv3 = _ConvBNReLU(in_channels=1024, out_channels= 512, kernel_size=3,
        #                           stride=1, padding=1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True)
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p = 0.3)
        self.sigmoid = nn.Sigmoid()
        #self.ppm = PyramidPooling(512, 512)
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv2d(512, 19, 1),
        )
        
    def forward(self, image, targets=None, flag = 'train'):
        size = image.size()[2:]
        lower_res = self.conv(image)
        #ppm_out = self.ppm(lower_res)
        #cat_out = lower_res + ppm_out
        #pool_out = self.avg(cat_out).squeeze()
        #if flag == 'train':
        #    intra_pairs, inter_pairs = self.get_pairs(pool_out)
        #    features1 = torch.cat([cat_out[intra_pairs[:,0],:,:,:], cat_out[inter_pairs[:,0],:,:,:]], dim = 0)
        #    features2 = torch.cat([cat_out[intra_pairs[:,1],:,:,:], cat_out[inter_pairs[:,1],:,:,:]], dim = 0)
        #    ppm_feat2 = torch.cat([ppm_out[intra_pairs[:,1],:,:,:], ppm_out[inter_pairs[:,1],:,:,:]], dim = 0)
        #    labels1 = torch.cat([targets[intra_pairs[:,0],:,:], targets[inter_pairs[:,0],:,:]], dim = 0)
        #    labels2 = torch.cat([targets[intra_pairs[:,1],:,:], targets[inter_pairs[:,1],:,:]], dim = 0)
        #    mut_feat = torch.cat([features1, features2], dim = 1)
        #    map_out = self.conv1(mut_feat)
                
            
        #    map_out = self.conv2(map_out)
        #    map_out = self.drop(self.conv3(map_out))
        #    gate1 = torch.mul(map_out, features1)
        #    gate1 = self.sigmoid(gate1)
            
        #    gate2 = torch.mul(map_out, features2)
        #    gate2 = self.sigmoid(gate2)
            
        #    features1_self = torch.mul(gate1, features1) + features1
        #    features1_self = self.classifier(features1_self)
        #    features1_self = F.interpolate(features1_self, size, mode='bilinear', align_corners=True)
        #    features1_other = torch.mul(gate2, features1) + features1
        #    features1_other = self.classifier(features1_other)
        #    features1_other = F.interpolate(features1_other, size, mode='bilinear', align_corners=True)
        #    features2_self = torch.mul(gate2, features2) + features2
        #    features2_self = self.classifier(features2_self)
        #    features2_self = F.interpolate(features2_self, size, mode='bilinear', align_corners=True)
        #    features2_other = torch.mul(gate1, features2) + features2
        #    features2_other = self.classifier(features2_other)
        #    features2_other = F.interpolate(features2_other, size, mode='bilinear', align_corners=True)
            
        #    return features1_self, features1_other, features2_self, features2_other, labels1, labels2
        
        #elif flag == 'eval':
        #    x = self.classifier(cat_out)
        #    x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        #    return tuple([x])
    """
    def get_pairs(self, embeddings):
        distance_matrix = pdist(embeddings).detach().cpu().numpy()
        num = embeddings.shape[0]
        intra_pairs = np.zeros([embeddings.shape[0], 2])
        inter_pairs  = np.zeros([embeddings.shape[0], 2])
        intra_pairs[:,0] = np.arange(num)
        inter_pairs[:,0] = np.arange(num)
        for i in range(num):
            dist = distance_matrix[i]
            sorted_dist = np.sort(dist)
            lowest = sorted_dist[1]
            sec_lowest = sorted_dist[2]
            intra_pairs[i, 1] = np.where(dist == lowest)[0][0]
            inter_pairs[i, 1] = np.where(dist == sec_lowest)[0][0]
        intra_pairs = torch.from_numpy(intra_pairs).long().to(device)
        inter_pairs = torch.from_numpy(inter_pairs).long().to(device)
        
        return intra_pairs, inter_pairs
     """   

def get_model(pretrained=False, root="./weights", **kwargs):
    model = my_model()
    return model
    