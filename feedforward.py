# feedforward.py
# CS 445/545 -- Spring 2020

# Class definition for a feedforward network
# in PyTorch.

# Imports
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Function to plot a single digit
def plot_digit(idx):
    img = np.squeeze(data[idx])
    fig = plt.figure(figsize = (12,12))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='Accent_r')
    ax.set_title(str(la_labels[idx].item()))
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y],2) if img[x][y] !=0 else 0
            ax.annotate(str(val), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='blue' if img[x][y]<thresh else 'red')

# Test out the plotting function
# idx = l_train[4][20]
# plot_digit(idx)

# This model is linear with 2 hidden layers using sigmoid activation function
class FeedForward(nn.Module):

    def __init__(self):
        super(Net,self).__init__() # This is required for initialization

        hid1_nodes = 1000
        hid2_nodes = 100

        self.ff1 = nn.Linear(28*28, hid1_nodes)
        self.ff2 = nn.Linear(hid1_nodes,hid2_nodes)
        self.ff3 = nn.Linear(hid2_nodes,10)

    def forward(self,x): # x will be each picture in 2D format 28 by 28 pixels
        x = x.view(-1,28*28) # Flatten the image
        x = F.sigmoid(self.ff1(x))
        x = F.sigmoid(self.ff2(x))
        x = self.ff3(x)
        return x