#read the image from csv, save to h5 as patch

import h5py
import csv
from collections import namedtuple
import numpy as np
import os
from PIL import Image
from sklearn.feature_extraction import image
from sklearn.feature_extraction.image import extract_patches
from blockwise_view import blockwise_view
from skimage.util.shape import view_as_windows

result = []
masks = []
img = []
idx = []
hw = []


def bbox1(img):
    a = np.where(img == 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox

patch_size = 256;
border = 72
base_folder = '/home/mb55411/dataset/splicing/NC2016_Test/'
count = 0

# image of 512x512

with open('val.csv') as f:
    csv_file = csv.reader(f,delimiter=',')
    headers = next(csv_file)
    Row = namedtuple('Row',headers)
    for r in csv_file:

        row = Row(*r)
        prob = Image.open(os.path.join(base_folder,row.ProbeFileName))
        mask = Image.open(os.path.join(base_folder,row.ProbeMaskFileName))
        #
        assert prob.size == mask.size
        width,height = prob.size

        mask = np.array(mask)
        prob = np.array(prob)
        
        ym,xm = mask.shape
        
        bbox = bbox1(mask)
        y_min,y_max,x_min,x_max = bbox

        if y_max - y_min < patch_size or x_max - x_min < patch_size:
            continue; 

        # enlarge bbox and get the area we want to analysis
        ymin = y_min - border if y_min - border > 0 else y_min
        xmin = x_min - border if x_min - border > 0 else x_min

        ymax = y_max + border if y_max + border < ym else y_max
        xmax = x_max + border if x_max + border < xm else x_max

        mask = np.ascontiguousarray(mask[ymin:ymax,xmin:xmax])
        prob = np.ascontiguousarray(prob[ymin:ymax,xmin:xmax,:])

        im = Image.fromarray(prob)
        mk = Image.fromarray(mask)

        im.save(os.path.join(base_folder,'fulls','images','val',str(count)+'.png'))
        mk.save(os.path.join(base_folder,'fulls','masks','val',str(count)+'.png'))
        count = count + 1

        print(row.ProbeFileID+'['+str(count)+']')     
        
        



