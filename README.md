# UNet-Pytorch
This repository contains a simple PyTorch implementation of U-Net [1] (with ResNet-50 [2] encoder) for multi-class image segmentation with custom datasets. My implementation is based on [this](https://github.com/usuyama/pytorch-unet) repository by [Usuyama](c), with some modifications to work with custom datasets.

## Dataset and data preprocessing
The dataset used in the repository is [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) [3], a 37 category pet dataset with roughly 200 images for each class. Because I just want to classify them as "dog" and "cat", I grouped pictures of cats together and did the same for dogs' pictures.
After that, I generated target masks by one-hot encoding the class labels - essentially creating an output channel for each of the possible class. In this case, there are no picures that contain both dog and cat, a mask for dogs in a picture of a cat will simply be a tensor filled with the scalar value 0.

## Create the UNet module
Instead of using the original implementation of UNet, I used [ResNet-50](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) backbone with pretrained weights. 

## Loss function
The chosen loss function is a combination of binary cross entropy and dice loss, which is used in the [original repository](https://github.com/usuyama/pytorch-unet). I also try focal loss but it doesn't improve the performance.

## Result
To be updated

## References
[1] Olaf Ronneberger, Philipp Fischer, Thomas Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. *arXiv:1505.04597, 2015*  
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. *arXiv:1512.03385, 2015*  
[3] Omkar M Parkhi and Andrea Vedaldi and Andrew Zisserman and C. V. Jawahar. The Oxford-IIIT Pet Dataset. *Retrieved from https://www.robots.ox.ac.uk/~vgg/data/pets/*
