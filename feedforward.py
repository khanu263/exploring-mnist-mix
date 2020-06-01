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
def plot_digit(idx: int, data: np.ndarray, la_labels: np.ndarray) -> None:
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

class FeedForward(nn.Module):
    '''
    This model is linear with 2 hidden layers using sigmoid activation function.
    Its initial input is a list of number of hidden nodes,
    for example to create a linear model with 3 hidden layers with respectively
    30, 150, 20 hidden nodes in them run the code FeedForward([30,150,20])
    '''
    def __init__(self, num_hid_nodes: list = [1000,100] ) -> None:
        super(FeedForward,self).__init__() # This is required for initialization
                
        num_hid_nodes.insert(0,(28*28)) # Add the number of inputs(pixels)
        num_hid_nodes.append(10) # Add the final number of outputs

        #self.layers will contain all the hidden number attribute names. e.g. ff1, ff2, ....
        self.layers = []
        for i in range(len(num_hid_nodes)-1):
            self.layers.append("ff"+str(i+1))
        
        for i,item in enumerate(self.layers):
            self.__dict__[item] = nn.Linear(num_hid_nodes[i], num_hid_nodes[i+1])

    def forward(self,x: np.ndarray) -> np.ndarray: # x will be each picture in 2D format 28 by 28 pixels
        x = x.view(-1,28*28) # Flatten the image
        for item in self.layers[:-1]:
            print(item)
            x = F.sigmoid(self.__dict__[item](x))
        x = self.__dict__[self.layers[-1]](x) # output doesn't need activation function
        return x