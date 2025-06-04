# Mitochondrial Segmentation from EM Images
This work takes the data from https://www.epfl.ch/labs/cvlab/data/data-em/ 
and performs semantic segmentation with the goal of creating masks for mitochondria.

####
There are examples outputs in the ```examples``` directory in this repo. The final test
accuracy after training was 0.9946, and the test IoU was 0.8850.

####
To run, simply clone this repo,
```commandline
git clone https://github.com/dwschaeper/Mitochondrial-Segmentation
```
and then run the main script. The first time it is run, the download flag should be used
but after that, refrain from using it more than necessary to not needlessly ping the host.
```commandline
python main.py --download
```
The final output images will be in the ```images``` directory.

## Usage
```commandline
usage: main.py [-h] [--download] [--patch_height PATCH_HEIGHT] [--patch_width PATCH_WIDTH] [--stride STRIDE] [--pairs PAIRS] [--background_proportion BACKGROUND_PROPORTION] [--background_augment]
               [--train_proportion TRAIN_PROPORTION] [--validation_proportion VALIDATION_PROPORTION] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--test_examples TEST_EXAMPLES]

Train a model to perform semantic segmentation from EM images and then can run the model.

options:
  -h, --help            show this help message and exit
  --download            Set this is you need to download the data. Please only perform this once as needed to not needless use the host site. (default: False)
  --patch_height PATCH_HEIGHT
                        The height of patches from the original (768, 1024) image. (default: 256)
  --patch_width PATCH_WIDTH
                        The width of patches from the original (768, 1024) image. (default: 256)
  --stride STRIDE       The stride when patching the large image determining how much overlap there is between patches. Default is no overlap. (default: 256)
  --pairs PAIRS         The number of image/mask pairs to plot during basic data exploration. (default: 5)
  --background_proportion BACKGROUND_PROPORTION
                        The proportion of the balanced dataset that should be background patches without mitochondria. If more background patches than what is available in the dataset are requested, the maximum available in     
                        the dataset will be used unless the '--background_augment' flag is used. (default: 0.2)
  --background_augment  Set this if you want to perform data augmentation on background patches to get to the proportion set in the '--background_proportion' parameter if there are not enough background patches in the dataset   
                        without augmentation. If there are enough, this will be ignored and no augmentation will be performed. These are simple rotations that will not leave a feature for the model to learn.This is enabled by   
                        default. (default: True)
  --train_proportion TRAIN_PROPORTION
                        The proportion of the total data that should be used for training. The testing proportionwill be 1 minus this value. (default: 0.8)
  --validation_proportion VALIDATION_PROPORTION
                        The proportion of the training split that will be reserved for validation. (default: 0.1)
  --batch_size BATCH_SIZE
                        The batch size to be used for training. (default: 32)
  --epochs EPOCHS       The number of training epochs. (default: 20)
  --test_examples TEST_EXAMPLES
                        The number of examples to depict from the test dataset. (default: 3)
```