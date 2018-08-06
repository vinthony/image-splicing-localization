# Image Splicing Localization Using CNN 

> This repo is part of project in Asia University Machine Learning Camp 2018


## What is Image Splicing

Spliced image is created from two authentic images. By masking the part of donor image, the selected region is pasted to the host image after some operations (translation and rescale the donor region). Sometimes, several post-processing techniques(such as Gaussian filter on the border of selected region) are used to the spliced region for the harmony of the selected region and host image.

![image](https://user-images.githubusercontent.com/4397546/43671765-04b292c2-97db-11e8-8709-e4097092302c.png)

##  Methods
As shown in bottom figure, We address the problem of ***image splicing localization***: given an input image, localizing the spliced region which is cut from another image. We formulate this as a classification task but, critically, instead of classifying the spliced region by local patch, we leverage the features from whole image and local patch together to classify patch. We call this structure Semi-Global Network. Our approach exploits the observation that the spliced region should not only highly relate to local features (spliced edges), but also global features (semantic information, illumination, etc.) from the whole image. We show that our method outperforms other state-of-the-art methods in Columbia datasets.


![image](https://user-images.githubusercontent.com/4397546/43671759-e8b2874e-97da-11e8-9f42-e2d0afe229bf.png)


## Installation

* Python 3.6
* PyTorch 0.3

You need to install the requirements by follow command firstly:

```shell
pip install -r requirements.txt
```

## Make Dataset

We use [Columbia Dataset](http://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/dlform.html) for training and testing. We Split all the spliced images(in subfolder `4cam_splc`) as three folds. Training(65%), validation(15%), testing(25%). For training faster, we firstly made patches dataset offline:

```shell
python tools/make_columbia_dataset /path/to/dataset
```
This script will generate image paches and resized full image for training and testing. So we have dataset: 

|| training| validation | testing |
|:---| :--: | :--: | :--: |
|patches| 14k | 3k| 5k|

## Training
You need to modify the parameters in shell script in `train_local.sh` for training the model, the full paramters list can be found in `hybird.py`:

```shell
python hybird.py\
  --epochs 60\
  --lr 1e-4\
  -c checkpoint/local\
  --arch sgn\
  --train-batch 64\
  --data columbia64\
  --base-dir /Users/oishii/Dataset/columbia/ 
```

## Watching

We use `TensorboardX`   to watch the training process, just install it by the [readme](https://github.com/lanpa/tensorboardX).

run the watching commond as :
```
tensorboard --logdir ./checkpoint
```

## Results

Here we show some sample results of our methods, from the left to right are the output of label, the output of mask and the ground truth mask:
![image](https://user-images.githubusercontent.com/4397546/43671725-4487e1fa-97da-11e8-8dad-e083ed1a9181.png)

Here are the label loss, segmentation loss, label accuracy, segmentation accuracy on validation set:
![image](https://user-images.githubusercontent.com/4397546/43671741-a03c7e20-97da-11e8-86b4-c6df5cb1b3c1.png)


## Acknowledgements

This work is partially support by Jeju National University and JDC (Jeju Free International City Development Center).


