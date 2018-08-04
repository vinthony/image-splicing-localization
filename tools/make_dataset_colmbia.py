#read the image from csv, save to h5 as patch

import csv,sys
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
data = 'columbia'+str(patch_size)
base_folder = sys.argv[1]


import csv
from collections import namedtuple
import random


original_images = []
train_rows = []
val_rows = []
test_rows = []
headers = []


original_images = [os.path.join(base_folder, '4cam_splc')+'/'+x for x in os.listdir(
    os.path.join(base_folder, '4cam_splc')) if x.endswith(".tif")]
print(len(original_images))

for v in original_images:
    rr = random.random()
    if rr<= 0.65:
        train_rows.append(v)
    elif rr <=0.75:
        val_rows.append(v)
    else:
        test_rows.append(v)


dic = {
    'train':train_rows,
    'val':val_rows,
    'test':test_rows
}

for str_type in ['train','val','test']:

    count = 0

    lst = []
    # mkdirs

    os.makedirs(os.path.join(base_folder,data,'images',str_type))
    os.makedirs(os.path.join(base_folder,data,'masks',str_type)) 
    os.makedirs(os.path.join(base_folder,data,'hole_probes',str_type)) 
    os.makedirs(os.path.join(base_folder,data,'resized_probes',str_type)) 


    for img_path in dic[str_type]: # each image in dic

        IMG_PATH = img_path
        MASK_PATH = img_path.replace(
            '4cam_splc', 'edgemask').replace('.tif', '_edgemask.jpg')

        prob = Image.open(IMG_PATH)
        rgbmask = Image.open(MASK_PATH) # here the mask is in rgb form, we need to change the non-spliced region to 255

        assert prob.size == rgbmask.size
        width,height = prob.size

        rgbmask = np.array(rgbmask)
        prob = np.array(prob)

        idx = np.argmax(rgbmask,2)

        mask = np.zeros([rgbmask.shape[0],rgbmask.shape[1]])
        mask[idx == 0] = 255 # rgb
        
        imdemo = Image.fromarray(mask)

        img_patches = blockwise_view(prob,(patch_size,patch_size,3),require_aligned_blocks=False)
        mask_patches = blockwise_view(mask,(patch_size,patch_size),require_aligned_blocks=False)

        img_patches = img_patches.squeeze(axis=2)


        for h in range(img_patches.shape[0]):
            for w in range(img_patches.shape[1]):
               
                im = Image.fromarray(img_patches[h][w])
                mk = Image.fromarray(mask_patches[h][w]).convert('RGB')

                # reshape the images from patches.
                #(h,w,y,x,3)
                tmp_patches = np.copy(img_patches)

                tmp_patches[h][w].fill(0.0)

                im_hole = tmp_patches.transpose(0, 2, 1, 3, 4).reshape(
                    (64*img_patches.shape[0], 64*img_patches.shape[1], 3))

                im_hole = Image.fromarray(im_hole)

                im_hole = im_hole.resize((224, 224), Image.ANTIALIAS)

                im_full = img_patches.transpose(0, 2, 1, 3, 4).reshape(
                    (64*img_patches.shape[0], 64*img_patches.shape[1], 3))
                im_full = Image.fromarray(im_full)
                im_full = im_full.resize((224, 224), Image.ANTIALIAS)

                PATCH_PATH = os.path.join(
                    data, 'images', str_type, str(count)+'.png')
                MASK_PATCH_PATH = os.path.join(
                    data, 'masks', str_type, str(count)+'.png')
                HOLE_IMAGE_PATH = os.path.join(
                    data, 'hole_probes', str_type, str(count)+'.png')
                FULL_IMAGE_PATH = os.path.join(
                    data, 'resized_probes', str_type, str(count)+'.png')

                im.save(base_folder+'/'+PATCH_PATH)
                mk.save(base_folder+'/'+MASK_PATCH_PATH)
                im_hole.save(base_folder+'/'+HOLE_IMAGE_PATH)
                im_full.save(base_folder+'/'+FULL_IMAGE_PATH)

                lst.append(",".join([PATCH_PATH, MASK_PATCH_PATH, FULL_IMAGE_PATH, MASK_PATH.replace(
                    base_folder, ''), IMG_PATH.replace(base_folder, ''), HOLE_IMAGE_PATH]))

                count = count + 1

        print('['+str(count)+']') 

    with open(os.path.join(base_folder,data,str_type+'.txt'),'w') as f:
        f.write("\n".join(lst))    
        
        



