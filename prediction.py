import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import  models
import torchvision.transforms as transforms

import os
import warnings
import argparse

import numpy as np

parser = argparse.ArgumentParser(description='CIFAR10 PyTorch inference')
parser.add_argument('--model_path', default='./cifar_model.pth', type=str, help='model output path')
args = parser.parse_args()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=8)

data_iter = iter(testloader)
images, labels = data_iter.next()
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = torch.load(args.model_path)

outputs = model(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for (inputs, labels) in testloader:
        if torch.cuda.is_available():
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on test dataset: %d %%' % (100 * correct / total))
