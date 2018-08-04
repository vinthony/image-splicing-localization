from __future__ import print_function, absolute_import

import os
import csv
import numpy as np
import json
import random
import math
import matplotlib.pyplot as plt
from collections import namedtuple
import h5py

import torch
import torch.utils.data as data

from scripts.utils.osutils import *
from scripts.utils.imutils import *
from scripts.utils.transforms import *
import torchvision.transforms as transforms

class Splicing(data.Dataset):
    def __init__(self, base_folder, img_folder, arch, patch_size=64):
        
        self.base_folder = base_folder
        self.patch_size = patch_size
        self.path = img_folder
        self.train = []
        self.anno = []
        self.img_fulls = [] # resized
        self.arch = arch
         # Data loading code
        with open(base_folder+img_folder) as f:
            for file_name in f.readlines():
                recoders = file_name.rstrip().split(',')
                # [PATCH_PATH,MASK_PATCH_PATH,FULL_IMAGE_PATH,MASK_PATH,IMG_PATH,HOLE_IMAGE_PATH])
                self.train.append(recoders[0])
                self.anno.append(recoders[1])
                self.img_fulls.append(recoders[2])
                
        print('total Dataset of '+img_folder+' is : ', len(self.train))


    def __getitem__(self, index):
        # different in tifs dataset and columbia dataset,
        # the mask in tifs dataset 1 represent image splicing region
        # the mask in columbia dataset 0 represent image splicing region
        
        img_path = self.base_folder + self.train[index]
        anno_path = self.base_folder + self.anno[index]

        img_full_path = self.base_folder + self.img_fulls[index]
        img_full = load_image(img_full_path)

        img  = load_image(img_path)  # CxHxW => automaticlly change 
        mask = load_image_gray(anno_path)

        if 'columbia' in self.path:
            mask = mask*-1 + 1
            
        # here 1 will represent image splicing region in mask

        label = 1 if mask.sum() > mask.numel() * 0.875  else  0
       
        return_tuple = (img,mask[0],label,img_full)

        return return_tuple

    def __len__(self):

        return len(self.train)
