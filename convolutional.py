# convolutional.py
# CS 445/545 -- Spring 2020

# Class definition for a convolutional network
# (ResNet) in PyTorch.

# Imports
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Basic building block of ResNet
class ResNetBasicBlock(nn.Module):

    # Initialization
    def __init__(self, in_planes, planes, stride = 1):

        # Superclass initialization
        super(ResNetBasicBlock, self).__init__()

        # Define the parts of the block
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)

        # The shortcut contains nothing
        self.shortcut = nn.Sequential()

        # If the structure isn't correct for empty shortcut, use 1x1 convolution
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size = 1, stride = stride, bias = False), nn.BatchNorm2d(planes))

    # Forward pass though block
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(self.bn2(out))
        return out

# Actual ResNet architecture
class ResNet(nn.Module):

    # Initialization
    def __init__(self, block, block_sizes, out_classes):

        # Superclass initialization
        super(ResNet, self).__init__()

        # First convolution
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(16)

        # ResNet blocks
        self.layer1 = self.make_layer(block, 16, block_sizes[0], stride = 1)
        self.layer2 = self.make_layer(block, 32, block_sizes[1], stride = 2)
        self.layer3 = self.make_layer(block, 64, block_sizes[2], stride = 2)

        # Final feedforward layer
        self.linear = nn.Linear(64, out_classes)

    # Make a ResNet block
    def make_layer(self, block, planes, num_blocks, stride):

        # Set up strides
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        # Make each block
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes

        # Return as a sequence of layers
        return nn.Sequential(*layers)

    # Forward pass through network
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer3(self.layer2(self.layer1(out)))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out