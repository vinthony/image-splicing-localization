# Image Splicing Localization Using CNN

We address the problem of ***image splicing localization***: given an input image, localizing the spliced region which is cut from another image. We formulate this as a classification task but, critically, instead of classifying the spliced region by local patch, we leverage the features from whole image and local patch together to classify patch. We call this structure Semi-Global Network. Our approach exploits the observation that the spliced region should not only highly relate to local features (spliced edges), but also global features (semantic information, illumination, etc.) from the whole image. Furthermore, we first integrate Fully Connected Conditional Random Fields as post-processing technique in image splicing to improve the consistency between the input image and the output of the network. We show that our method outperforms other state-of-the-art methods in three popular datasets.

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

Here we show some sample results of our methods:









