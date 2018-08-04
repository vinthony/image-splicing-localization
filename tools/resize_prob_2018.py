
import numpy as np
import os
from PIL import Image

base = '/home/mb55411/dataset/nc2018/MFC18_Dev1_Ver1/'

with open(base+'nc2018_png.images') as f:
    for file_name in f.readlines():
        img = Image.open(file_name.rstrip()).convert('RGB')
        img = img.resize((256,256),Image.ANTIALIAS)
        img.save(base+'resized_probes/'+file_name.rstrip().split('/')[-1])

with open(base+'nc2018_jpg.images') as f:
    for file_name in f.readlines():
        img = Image.open(file_name.rstrip()).convert('RGB')
        img = img.resize((256,256),Image.ANTIALIAS)
        img.save(base+'resized_probes/'+file_name.rstrip().split('/')[-1])