# Train Basic Model on CIFAR10-Dataset

<p align="center">
<img src="https://user-images.githubusercontent.com/59862546/117540979-59b64100-b02f-11eb-9ea9-457ecf2e2271.png" width="400" height="200">    <img src="https://user-images.githubusercontent.com/16641054/46775076-8b17e480-cd40-11e8-9501-89c6fbca36bd.jpg" width="400" height="200"> 
<p>
  
## Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Training](#training)

## Introduction
The `CIFAR-10` dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

below is the  6 random images with their respective label:

<img src="https://miro.medium.com/max/1182/1*OSvbuPLy0PSM2nZ62SbtlQ.png" width="500" height="250">

There is a package of python called `torchvision`, that has data loaders for `CIFAR10` and data transformers for images using `torch.utils.data.DataLoader`.

Below an example of how to load `CIFAR10` dataset using `torchvision`:

```python
import torch
import torchvision
## load data CIFAR10
train_dataset = torchvision.datasets.CIFAR10(root='./train_data', train=True, download=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
```

## Prerequisites
- Python>=3.6
- PyTorch >=1.4
- Library are mentioned in `requirenments.txt`

## Training
I used pretrained `resnet18` for model training. you can use any other pretrained model according to you problem.
```python
import torchvision.models as models
alexnet = models.alexnet()
vgg16 = models.vgg16()
densenet = models.densenet161()
inception = models.inception_v3()
```
There are two things for pytorch model training:
1. Notebook - you can just download and play with it
2. python scripts:
    ```
    # Start training with: 
    python main.py
    
    # You can manually pass the attributes for the training: 
    python main.py --lr=0.01 --epoch 20 --model_path './cifar_model.pth'
    
    # Start infrence with:
     python3.6 prediction.py --model_path './cifar_model.pth'
    ```


# Give a :star: to this Repository!
