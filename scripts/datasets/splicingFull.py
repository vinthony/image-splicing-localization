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
from tools.blockwise_view import blockwise_view
from PIL import Image


class SplicingFull(data.Dataset):
    def __init__(self, base_folder, img_folder, arch, patch_size=64):

        self.base_folder = base_folder
        self.patch_size = patch_size
        self.path = img_folder
        self.train = []
        self.anno = []
        self.original_image = []
        self.original_mask = []
        self.img_fulls = []  # resized
        self.arch = arch
        # Data loading code
        with open(base_folder+img_folder) as f:
            for file_name in f.readlines():
                recoders = file_name.rstrip().split(',')
                # [PATCH_PATH,MASK_PATCH_PATH,FULL_IMAGE_PATH,MASK_PATH,IMG_PATH,HOLE_IMAGE_PATH])
                # self.train.append(recoders[0])
                # self.anno.append(recoders[1])
                # self.img_fulls.append(recoders[2])
                if recoders[4] not in self.original_image:
                    self.original_image.append(recoders[4])
                    if 'columbia' in self.path:
                        self.original_mask.append(recoders[3])
                    else:
                        self.original_mask.append(recoders[3].replace(
                        'DSO-1', 'DSO-1-Fake-Images-Masks'))

        print('total valization of '+img_folder +
              ' is : ', len(self.original_image))

    def __getitem__(self, index):

        img_path = self.base_folder + self.original_image[index]
        anno_path = self.base_folder + self.original_mask[index]

        # print(img_path,anno_path)

        # img = load_image(img_path)  # CxHxW => automaticlly change
        # mask = load_image_gray(anno_path)
        img = Image.open(img_path).convert('RGB')
        img_full = img.resize((224,224),Image.ANTIALIAS)
        
        mask = Image.open(anno_path)
        mask = mask.resize(img.size,Image.ANTIALIAS)
        
        img = np.array(img).astype(np.float32)/255  # [0,1】

        if 'columbia' in self.path: # we don't need to process tifs dataset here.
            # convert rgb [red as 1]
            rgbmask = mask.convert('RGB')
            rgbmask = np.array(rgbmask)
            idx = np.argmax(rgbmask,2) 
            mask = np.ones([rgbmask.shape[0],rgbmask.shape[1]])
            mask[idx == 0] = 0 # red
        else:
            mask = mask.convert('L')
            mask = np.array(mask).astype(np.float32)/255  # [0,1]
            mask = mask*-1 + 1
            

#        8*8*64*64*3
        image_patches = blockwise_view(
        np.array(img), (64, 64, 3), require_aligned_blocks=False).squeeze(axis=2)
        mask_patches = blockwise_view(
            np.array(mask), (64, 64), require_aligned_blocks=False).squeeze()

        mask = mask[0:64*image_patches.shape[0],0:64*image_patches.shape[1],np.newaxis]
        
        batchsize = image_patches.shape[0]*image_patches.shape[1]
        image_patches = np.reshape(image_patches, (batchsize, 64, 64, 3))
        mask_patches = np.reshape(mask_patches,(batchsize, 64,64))
        
        img_full = im_to_torch(img_full).unsqueeze(0)
        # print(img_full)
        img_full = img_full.repeat(batchsize,1,1,1)

        mask = im_to_torch(mask)[0]
        # patches to torch
        image_patches = to_torch(np.transpose(
            image_patches, axes=(0, 3, 1, 2)))

        labels = np.zeros((batchsize))

        threshold = 64*64*7/8

        for i in range(batchsize):
            if np.sum(mask_patches[i]) > threshold:  # 64*64*7/8
                labels[i] = 1

        return (image_patches, labels, mask, img_full)

    def __len__(self):

        return len(self.original_image)
