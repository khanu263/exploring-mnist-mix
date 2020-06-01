# feedforward.py
# CS 445/545 -- Spring 2020

# Class definition for a feedforward network
# in PyTorch.

# Imports
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Function to plot a single digit
def plot_digit(idx: int, data: np.ndarrey, la_labels: np.ndarrey) -> None:
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

    def __init__(self, num_hid_nodes = [1000,100]: list) -> None:
        super(FeedForward,self).__init__() # This is required for initialization
                
        num_hid_nodes.insert(0,(28*28)) # Add the number of inputs(pixels)
        num_hid_nodes.append(10) # Add the final number of outputs
        
        #self.layers will contain all the hidden number attribute names. e.g. ff1, ff2, ....
        self.layers = []
        for i in range(len(num_hid_nodes)+1):
            self.layers.append("ff"+str(i+1))
        
        for item in self.layers:
            self.__dict__[item] = nn.Linear(num_hid_nodes[i], num_hid_nodes[i+1])

    def forward(self,x: np.ndarrey) -> np.ndarrey: # x will be each picture in 2D format 28 by 28 pixels
        x = x.view(-1,28*28) # Flatten the image
        for item in self.layers[:-1]:
            x = F.sigmoid(self.__dict__[item](x))
        x = self.__dict__[self.layers[-1]](x) # output doesn't need activation function
        return x