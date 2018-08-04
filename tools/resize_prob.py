
import numpy as np
import os
from PIL import Image

with open('/home/mb55411/dataset/tifs-database/www.images') as f:
    for file_name in f.readlines():
        img = Image.open(file_name.rstrip().replace('images','probes'))
        img = img.resize((256,256),Image.ANTIALIAS)
        print('/home/mb55411/dataset/tifs-database/resized_probes/'+file_name.rstrip().split('/')[-1])
        img.save('/home/mb55411/dataset/tifs-database/resized_probes/'+file_name.rstrip().split('/')[-1])

