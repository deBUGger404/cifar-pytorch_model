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

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='CIFAR10 PyTorch training')
parser.add_argument('--lr', default=0.1, type=float, help='learning-rate')
parser.add_argument('--epochs', default=5, type=int, help='number of epoch')
parser.add_argument('--model_path', default='./cifar_model.pth', type=str, help='model output path')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


net = models.resnet18(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Sequential(nn.Linear(num_ftrs, 256),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(256, len(classes)))

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

for epoch in range(args.epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        if torch.cuda.is_available():
            inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    print('[%d, %5d] loss: %.3f Acc: %.3f' %
                  (epoch + 1, i + 1, running_loss / len(trainloader),100.*correct/total))
    running_loss = 0.0

print('Finished Training')

### model save
torch.save(net, args.model_path)
