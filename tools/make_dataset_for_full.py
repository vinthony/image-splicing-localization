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

patch_size = 100;
# border = 48;
base_folder = '/home/mb55411/dataset/splicing/NC2016_Test/'
count = 0

lst = []

with open('test.csv') as f:
    csv_file = csv.reader(f,delimiter=',')
    headers = next(csv_file)
    Row = namedtuple('Row',headers)
    for r in csv_file:

        row = Row(*r)
        if row.TaskID == "Splice":
            prob = Image.open(os.path.join(base_folder,row.ProbeFileName))
            mask = Image.open(os.path.join(base_folder,row.ProbeMaskFileName))
            #
            assert prob.size == mask.size
            width,height = prob.size

            mask = np.array(mask)
            prob = np.array(prob)
            
            img_prob = Image.fromarray(prob)


            img_patches = blockwise_view(prob,(patch_size,patch_size,3),require_aligned_blocks=False)
            mask_patches = blockwise_view(mask,(patch_size,patch_size),require_aligned_blocks=False)

            img_patches = img_patches.squeeze(axis=2)

            for h in range(img_patches.shape[0]):
                for w in range(img_patches.shape[1]):
                   
                    im = Image.fromarray(img_patches[h][w])
                    mk = Image.fromarray(mask_patches[h][w])

                    im.save(os.path.join(base_folder,'splicing100','images','test',str(count)+'.png'))
                    mk.save(os.path.join(base_folder,'splicing100','masks','test',str(count)+'.png'))
                    lst.append(os.path.join(base_folder,row.ProbeFileName))
                    count = count + 1

            print(row.ProbeFileID+'['+str(count)+']') 

    with open(os.path.join(base_folder,'splicing100','test.txt'),'w') as f:
        f.write("\n".join(lst))    
        
        



