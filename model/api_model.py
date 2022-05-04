#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:57:11 2021

@author: rajiv
"""

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
from torchvision import models

device = 'cuda'

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

"convolution - BatchBorn - Leaky ReLU"
class conv_bn_lrelu(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(conv_bn_lrelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(True)
        )

    def forward(self, x):
        return self.conv(x)
    
" Sep Convolution - BatchNorm - Leaky ReLU"
class dept_sep_conv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(dept_sep_conv, self).__init__()
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

"Decomposed Transposed Conv - BatchNorm - Leaky ReLU"
class dec_tra_conv(nn.Module):
    """Decomposed transposed convolutions"""
    def __init__(self, dw_channels, out_channels, kernel_size=5, padding=1, stride=2):
        super(dec_tra_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(dw_channels, dw_channels,
                               groups = dw_channels, kernel_size = kernel_size,
                               padding=padding, stride = stride, bias = False),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(dw_channels, out_channels, 1, bias = False),
            nn.LeakyReLU(True)
            )
    def forward(self, x):
        return self.conv(x)
    
"Downsampling module"
class learning_to_downsample(nn.Module):
    def __init__(self, channel1 = 32, channel2 = 48, out_channel = 64, **kwargs):
        super(learning_to_downsample, self).__init__()
        self.conv = conv_bn_lrelu(in_channels=3, out_channels=channel1,
                                  kernel_size=3, stride = 2, padding = 1)
        self.sep_conv1 = dept_sep_conv(dw_channels = channel1, out_channels=channel2,
                                      kernel_size = 2)
        self.sep_conv2 = dept_sep_conv(dw_channels = channel2, out_channels=out_channel,
                                       kernel_size = 2, padding = 1)
    def forward(self, x):
        x = self.conv(x)
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        return x
    
class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = dept_sep_conv(dw_channels, dw_channels, stride)
        self.dsconv2 = dept_sep_conv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1),
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x

    
class decoder(nn.Module):
    def __init__(self, channel1 = 1024, channel2= 512, channel3= 64, **kwargs):
        super(decoder, self).__init__()
        self.tconv1 = dec_tra_conv(dw_channels= 512, out_channels = channel1, kernel_size=4, padding=1, stride=2)
        self.tconv2 = dec_tra_conv(dw_channels= channel1, out_channels=channel2, kernel_size = 4, padding=1, stride=2)
        self.tconv3 = dec_tra_conv(dw_channels= channel2, out_channels=channel3, kernel_size = 4, padding=1, stride=2)
    def forward(self, x):
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = F.interpolate(x, (128,128), mode = 'bilinear', align_corners=True)
        return x


class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()
        self.lds = learning_to_downsample(32, 48, 64)
        layers = models.resnet18(pretrained=True)
        layers = list(layers.children())
        self.conv = nn.Sequential(*layers[:-2])
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv1 = conv_bn_lrelu(in_channels=1024, out_channels= 1792, kernel_size=3,
                                   stride=1, padding=1)
        self.conv2 = conv_bn_lrelu(in_channels=1792, out_channels= 1024, kernel_size=3,
                                   stride=1, padding=1)
        self.conv3 = conv_bn_lrelu(in_channels=1024, out_channels= 512, kernel_size=3,
                                   stride=1, padding=1)
        self.drop = nn.Dropout(p = 0.3)
        self.sigmoid = nn.Sigmoid()
        self.decoder = decoder(256, 128, 64)
        self.classifier = Classifer(64, 19)
        
    def forward(self, image, targets=None, flag = 'train'):
        size = image.size()[2:]
        downsized = self.lds(image)
        lower_res = self.conv(image)
        pool_out = self.avg(lower_res).squeeze()
        if flag == 'train':
            intra_pairs, inter_pairs = self.get_pairs(pool_out)
            features1 = torch.cat([lower_res[intra_pairs[:,0],:,:,:], lower_res[inter_pairs[:,0],:,:,:]], dim = 0)
            downsized_feature1 = torch.cat([downsized[intra_pairs[:,0]], downsized[inter_pairs[:,0]]], dim = 0)
            features2 = torch.cat([lower_res[intra_pairs[:,1],:,:,:], lower_res[inter_pairs[:,1],:,:,:]], dim = 0)
            downsized_feature2 = torch.cat([downsized[intra_pairs[:, 1]], downsized[inter_pairs[:, 1]]], dim = 0)
            labels1 = torch.cat([targets[intra_pairs[:,0],:,:], targets[inter_pairs[:,0],:,:]], dim = 0)
            labels2 = torch.cat([targets[intra_pairs[:,1],:,:], targets[inter_pairs[:,1],:,:]], dim = 0)
            mut_feat = torch.cat([features1, features2], dim = 1)
            map_out = self.conv1(mut_feat)
                
            
            map_out = self.conv2(map_out)
            map_out = self.drop(self.conv3(map_out))
            gate1 = torch.mul(map_out, features1)
            gate1 = self.sigmoid(gate1)
            
            gate2 = torch.mul(map_out, features2)
            gate2 = self.sigmoid(gate2)
            
            features1_self = torch.mul(gate1, features1) + features1
            features1_self = self.decoder(features1_self)
            features1_self = features1_self + downsized_feature1
            features1_self = F.interpolate(features1_self, (256,256), mode='bilinear', align_corners=True)
            features1_self = self.classifier(features1_self)
            features1_other = torch.mul(gate2, features1) + features1
            features1_other = self.decoder(features1_other)
            features1_other = features1_other + downsized_feature1
            features1_other = F.interpolate(features1_other, (256,256), mode='bilinear', align_corners=True)
            features1_other = self.classifier(features1_other)
            features2_self = torch.mul(gate2, features2) + features2
            features2_self = self.decoder(features2_self)
            features2_self = features2_self + downsized_feature2
            features2_self = F.interpolate(features2_self, (256,256), mode='bilinear', align_corners=True)
            features2_self = self.classifier(features2_self)
            features2_other = torch.mul(gate1, features2) + features2
            features2_other = self.decoder(features2_other)
            features2_other = features2_other + downsized_feature2
            features2_other = F.interpolate(features2_other, (256,256), mode='bilinear', align_corners=True)
            features2_other = self.classifier(features2_other)
            
            return features1_self, features1_other, features2_self, features2_other, labels1, labels2
        
        elif flag == 'eval':
            x = self.decoder(lower_res)
            x = F.interpolate(x, (512,1024), mode='bilinear', align_corners=True)
            x = x + downsized
            x = F.interpolate(x, size, mode='bilinear', align_corners=True)
            x = self.classifier(x)
            return tuple([x])
    
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
        

def get_model(pretrained=False, root="./weights", **kwargs):
    if pretrained:
        model = my_model()
        model.load_state_dict(torch.load(root+"/fast_scnn_citys.pth"))
    else:
        model = my_model()
    return model
    