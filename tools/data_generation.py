import sys
sys.path.append('..')
from pycocotools.coco import COCO
import numpy as np
import pylab
from PIL import Image
import random
import time

SUN2012_dirs = '../../sun2012.images'

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
data_dir = '..'
data_type = 'train2014'

ann_file = '{}/annotations/instances_{}.json'.format(data_dir,data_type)

output_dir = './train/images/'
mask_dir = './train/annos/'

coco = COCO(ann_file)
## get all the categories
total_need_to_generate = 30000
count_of_valid_imgs = 0

cat_ids = coco.getCatIds()
img_ids = coco.getImgIds()


def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None,expand=False):
    if center is None:
        return image.rotate(angle)
    angle = -angle/180.0*math.pi
    nx,ny = x,y = center
    sx=sy=1.0
    if new_center:
        (nx,ny) = new_center
    if scale:
        (sx,sy) = scale
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine/sx
    b = sine/sx
    c = x-nx*a-ny*b
    d = -sine/sy
    e = cosine/sy
    f = y-nx*d-ny*e
    return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=Image.BICUBIC)

## read SUN 2012

sun2012_img_list = [] 
with open(SUN2012_dirs) as f:
	for img in f:
		sun2012_img_list.append(img.rstrip())

while 1:
	if count_of_valid_imgs > total_need_to_generate: break
	# read image from sun2012 randomly.
	sun2012_path = sun2012_img_list[np.random.randint(0,len(sun2012_img_list))];
	imA = Image.open(sun2012_path).convert('RGB') 
	## image transformation for sun2012 image.

	minimum_square = min(imA.width,imA.height);

	borderY = (imA.height-minimum_square)//2; borderX = (imA.width-minimum_square)//2;
	
	x_coor,y_coor = borderX *random.random(), borderY *random.random()


	imA=imA.crop((x_coor,y_coor,x_coor+minimum_square,y_coor+minimum_square))

	imA=imA.resize((256,256),Image.BILINEAR)

	imA_original = imA.copy()

	# generate the mask from mscoco.
	catIds = cat_ids[np.random.randint(0,len(cat_ids))];
	imgIds = coco.getImgIds(catIds=catIds);
	img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
	annids = coco.getAnnIds(imgIds=img['id'],catIds=catIds,iscrowd=None)
	anns = coco.loadAnns(annids)[0];

	# ignore the small area.
	if anns['area'] < 1000: continue;
	
	# crop the image from BBOX.
	bbox = anns['bbox'];
	bbox_r = max(bbox[2],bbox[3])//1.5;

	center_of_bbox_x,center_of_bbox_y = bbox[2]//2+bbox[0], bbox[3]//2+bbox[1]

	bbox_new = (center_of_bbox_x-bbox_r,center_of_bbox_y-bbox_r,center_of_bbox_x+bbox_r,center_of_bbox_y+bbox_r)

	re = coco.annToMask(anns)
	mask = np.concatenate((re[...,np.newaxis],re[...,np.newaxis],re[...,np.newaxis]),axis=2)


	mask = Image.fromarray(mask.astype('uint8'), 'RGB')
	imB = Image.open('%s/images/%s/%s'%(data_dir,data_type,img['file_name'])).convert('RGB')


	# if imB.mode != 'RGB':
	# 	imB = np.concatenate((imB[...,np.newaxis],imB[...,np.newaxis],imB[...,np.newaxis]),axis=2)

	# if imA.mode != 'RGB':
	# 	imA = np.concatenate((imA[...,np.newaxis],imA[...,np.newaxis],imA[...,np.newaxis]),axis=2)

	# so we need to find the (256,256) box of imb and save the image.
	imB = imB.crop(bbox_new);
	mask = mask.crop(bbox_new);


	imB =  imB.resize((256,256),Image.BILINEAR)
	mask = mask.resize((256,256),Image.BILINEAR)

	imB_original = imB.copy()
	mask_original =  Image.fromarray(np.copy(mask).astype('uint8'),'RGB') 

	angle = random.random()*20-10;
	scale = random.random()*4+0.5;

	imB = ScaleRotateTranslate(imB,angle,scale=scale)
	mask = ScaleRotateTranslate(mask,angle,scale=scale)


	imB = imB.resize((256,256),Image.BILINEAR)
	mask = mask.resize((256,256),Image.BILINEAR)

	mask_background = Image.fromarray( np.zeros((256,256,3), dtype=np.uint8),'RGB')
	reverse_background = Image.fromarray( np.ones((256,256,3), dtype=np.uint8),'RGB')

	splicing_image = Image.fromarray(np.multiply(imB,mask).astype('uint8'), 'RGB')

	# current size of splicing.
	offsetx = np.random.randint(-127,127);
	offsety = np.random.randint(-127,127);
	box = tuple( ( int(offsetx),int(offsety),256+int(offsetx),256+int(offsety) ));

	reverse_mask =  Image.fromarray(np.multiply( np.add(mask,-1) ,-1).astype('uint8'), 'RGB');
	reverse_img = Image.fromarray(np.multiply(imA.crop(box),reverse_mask).astype('uint8'), 'RGB')

	mixed_area = Image.fromarray(( np.add(splicing_image,reverse_img)).astype('uint8'), 'RGB')

	imA.paste(mixed_area,box)
	mask_background.paste(mask_original,box)
	reverse_background.paste(reverse_mask,box)

	total_width = 256*2
	max_height = 256


	mask_background = Image.fromarray(np.multiply(mask_background,255).astype('uint8'), 'RGB');
	reverse_background = Image.fromarray(np.multiply(reverse_background,255).astype('uint8'), 'RGB');
	mask_original = Image.fromarray(np.multiply(mask_original,255).astype('uint8'), 'RGB')

	type1_img = Image.new('RGB', (total_width, max_height))
	type1_img.paste(imA_original,(0,0))
	type1_img.paste(imB,(256,0))

	type1_mask = Image.new('RGB', (total_width, max_height))
	type1_mask.paste( (0,0,0), [0,0,type1_mask.size[0],type1_mask.size[1]])


	type2_img = Image.new('RGB', (total_width, max_height))
	type2_img.paste(imA_original,(0,0))
	type2_img.paste(imA,(256,0))


	type2_mask = Image.new('RGB', (total_width, max_height))
	type2_mask.paste(reverse_background,(0,0))
	type2_mask.paste(reverse_background,(256,0))

	type3_img = Image.new('RGB', (total_width, max_height))
	type3_img.paste(imA,(0,0))
	type3_img.paste(imB_original,(256,0))

	type3_mask = Image.new('RGB', (total_width, max_height))
	type3_mask.paste(mask_background,(0,0))
	type3_mask.paste(mask_original,(256,0))


	## save the images:
	img_name = str(time.time());
	type1_img.save(output_dir+img_name+'_1.jpg')
	type2_img.save(output_dir+img_name+'_2.jpg')
	type3_img.save(output_dir+img_name+'_3.jpg')

	type1_mask.save(mask_dir+img_name+'_1.jpg')
	type2_mask.save(mask_dir+img_name+'_2.jpg')
	type3_mask.save(mask_dir+img_name+'_3.jpg')



	count_of_valid_imgs = count_of_valid_imgs + 1
	if count_of_valid_imgs % 100 == 0:
		print('generate {} | {} '.format(count_of_valid_imgs,total_need_to_generate))


	

	