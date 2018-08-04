
import numpy as np
import os
from PIL import Image

with open('/home/mb55411/dataset/nc2017/NC2017_Dev_Ver1_Img/nc2017.images') as f:
    for file_name in f.readlines():
        img = Image.open(file_name.rstrip()).convert('RGB')
        img = img.resize((256,256),Image.ANTIALIAS)
        img.save('/home/mb55411/dataset/nc2017/NC2017_Dev_Ver1_Img/resized_probes/'+file_name.rstrip().split('/')[-1])