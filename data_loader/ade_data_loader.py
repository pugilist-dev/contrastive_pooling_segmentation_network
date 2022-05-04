#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 01:18:50 2021

@author: rajiv
"""

import os
import random
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image, ImageOps, ImageFilter


class ade_segmentation(data.Dataset):
    
    NUM_CLASSES = 19
    
    def __init__(self, root='./ADE20K_2016_07_26', split='training', mode = None,
                 transform=None, base_size = 480, crop_size = 256, **kwargs):
        super(ade_segmentation, self).__init__()
        self.root = root
        self.split = split
        self.mode = mode if mode is not None else split
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size
        self.img_path, self.mask_path = _get_ade_pairs(self.root, self.split)
        assert (len(self.img_path) == len(self.mask_path))
        
    

def _get_ade_pairs(folder, split='training'):
    def get_ade_pairs(path):
        img_list = []
        mask_list = []
        for folder_1 in os.listdir(path):
            for folder_2 in os.listdir(path+"/" + folder_1):
                for root, _, files in os.walk(path + "/" + folder_1 + "/" + folder_2):
                    files = sorted(files)
                    for filename in files:
                        if filename.endswith('.jpg'):
                            imgpath = path + "/" + folder_1 + "/" + folder_2 + "/" + filename
                            img_list.append(imgpath)
                        elif filename.endswith('seg.png'):
                            maskpath = path + "/" + folder_1 + "/" + folder_2 + "/" + filename
                            mask_list.append(maskpath)
        assert (len(img_list) == len(mask_list))
        if len(img_list) == 0:
            raise RuntimeError("Found 0 input images in folder" + path + "\n")
        elif len(mask_list) == 0:
            raise RuntimeError("Found 0 mask images in folder" + path + "\n")
        return img_list, mask_list
    
    if split in ('training'):
        img_folder = os.path.join(folder, 'images', split)
        img_path, mask_path = get_ade_pairs(img_folder)
        return img_path, mask_path
    elif split in('validation'):
        img_folder = os.path.join(folder, 'images', split)
        img_path, mask_path = get_ade_pairs(img_folder)
        return img_path, mask_path
        
        
    