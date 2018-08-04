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

patch_size = 64;
base_folder = '/home/mb55411/dataset/nc2017/NC2017_Dev_Ver1_Img'


for str_type in ['train','val','test']:

    count = 0

    lst = []
    # mkdirs

    os.makedirs(os.path.join(base_folder,'patches','images',str_type))
    os.makedirs(os.path.join(base_folder,'patches','masks',str_type)) 

    print('processing:'+'mfc2017_'+str_type+'.csv')

    with open('mfc2017_'+str_type+'.csv') as f:
        csv_file = csv.reader(f,delimiter=',')
        Row = namedtuple('Row',next(csv_file))

        for r in csv_file:

            row = Row(*r)
            if row.TaskID == "splice" and row.ProbeFileName.find('jpg') > -1:

                IMG_PATH = os.path.join(base_folder,row.ProbeFileName)
                MASK_PATH = os.path.join(base_folder,row.BinaryProbeMaskFileName)

                prob = Image.open(IMG_PATH)
                mask = Image.open(MASK_PATH)

                assert prob.size == mask.size
                width,height = prob.size

                mask = np.array(mask)
                prob = np.array(prob)
                
                img_patches = blockwise_view(prob,(patch_size,patch_size,3),require_aligned_blocks=False)
                mask_patches = blockwise_view(mask,(patch_size,patch_size),require_aligned_blocks=False)

                img_patches = img_patches.squeeze(axis=2)

                for h in range(img_patches.shape[0]):
                    for w in range(img_patches.shape[1]):
                       
                        im = Image.fromarray(img_patches[h][w])
                        mk = Image.fromarray(mask_patches[h][w])

                        PATCH_PATH = os.path.join(base_folder,'patches','images',str_type,str(count)+'.png')
                        MASK_PATCH_PATH = os.path.join(base_folder,'patches','masks',str_type,str(count)+'.png')

                        im.save(PATCH_PATH)
                        mk.save(MASK_PATCH_PATH)
                        lst.append(",".join([PATCH_PATH,MASK_PATCH_PATH,IMG_PATH,MASK_PATH]))
                        count = count + 1

                print(row.ProbeFileID+'['+str(count)+']') 

        with open(os.path.join(base_folder,'patches',str_type+'.txt'),'w') as f:
            f.write("\n".join(lst))    
        
        



