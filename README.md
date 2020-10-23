# UNet-Pytorch
This repository contains simple PyTorch implementations of U-Net[1] (with ResNet-50 encoder) for multi-class image segmentation with custom dataset. My implementation is based on [this](https://github.com/usuyama/pytorch-unet)[2] repository by [Usuyama](https://github.com/usuyama), with some modification to works with custom datasets.

## Dataset and data preprocessing
The dataset used in the repository is [The Oxford-IIIT Pet Dataset](https://www.kaggle.com/tanlikesmath/the-oxfordiiit-pet-dataset)[3], a 37 category pet dataset with roughly 200 images for each class. Because I just want to classify them as "dog" and "cat", I grouped pictures of cats together and do the same for dogs' pictures.
After that, I generated target masks by one-hot encoding the class labels - essentially creating an output channel for each of the possible classes. In this case, there are no picures that contain both dog and cat, a mask for dogs in a picture of a cat will simply be a tensor filled with the scalar value 0.

## Create the UNet module
Instead of using the original implementation of UNet, I used [ResNet-50](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) backbone with pretrained weights. 

## Loss function
The chosen loss function is a combination of binary cross entropy and dice loss, which is used in the [original repository](https://github.com/usuyama/pytorch-unet). I also try focal loss but it doesn't improve the performace.

## Result
To be updated
