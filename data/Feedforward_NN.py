# preprocess.py
# CS 445/545 -- Spring 2020

# Use this script with the original MNIST-MIX data to
# save a NumPy array containing all examples and a list
# of split files describing indices and classes for
# the various experiments.

# Usage: place this in the same directory with X_train.npy,
#        y_train.py, X_test.npy, and y_test.npy, then run.

# Imports
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Load raw data
dataset = 'MNIST_MIX_train_test.npz'
data = np.load(dataset)
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
print("Loaded raw data.")

# Create full dataset and scale
data = np.concatenate((X_train, X_test))
data = data / 255
print("Created and scaled full dataset.")

# Create full sets of labels
ls_labels = np.concatenate((y_train, y_test)).astype(np.int32)
la_labels = np.mod(ls_labels, 10).astype(np.int32)
print("Created language-specific and language-agnostic labels.")

# Initialize train and test splits
cutoff = len(X_train)
end = len(data)
all_train = []
all_test = []
l_train = []
l_test = []
# Creates a list of 10 empty lists
for i in range(10):
    l_train.append([])
    l_test.append([])

# Go through data and assign indices
for i in range(cutoff):
    all_train.append(i) # sequential list of indices
    l_train[ls_labels[i] // 10].append(i) # list of indices for each language, separately  
for i in range(cutoff, end):
    all_test.append(i) # sequential list of indices
    l_test[ls_labels[i] // 10].append(i) # list of indices for each language, separately  
print("Created train/test splits.")

# Define number-language relation
languages = {0: "arabic",
             1: "bangla",
             2: "devanagari",
             3: "english",
             4: "farsi",
             5: "kannada",
             6: "swedish",
             7: "telegu",
             8: "tibetan",
             9: "urdu"}

# Save arrays
np.save("data.npy", data)
print("Saved data array.")
np.save("ls_labels.npy", ls_labels)
np.save("la_labels.npy", la_labels)
print("Saved label arrays.")

# Make split directory if it doesn't exist
if not os.path.exists("splits"):
    os.makedirs("splits")

# Save all-data split
with open("splits/all_train.split", "w") as f:
    f.writelines("%d\n" % idx for idx in all_train)
with open("splits/all_test.split", "w") as f:
    f.writelines("%d\n" % idx for idx in all_test)
print("Saved all-data splits.")

# Save per-language split
for i in range(10):
    with open("splits/{}_{}_train.split".format(i, languages[i]), "w") as f:
        f.writelines("%d\n" % idx for idx in l_train[i])
    with open("splits/{}_{}_test.split".format(i, languages[i]), "w") as f:
        f.writelines("%d\n" % idx for idx in l_test[i])
print("Saved per-language splits.")

for i in range(10):
    print("Test portion of data for the language %s: %.2f %%" %(languages[i],(len(l_test[i])/(len(l_test[i])+len(l_train[i])))*100))
    
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
idx = l_train[4][20]
plot_digit(idx)

# This model is linear with 2 hidden layers using sigmoid activation function
class Net(nn.Module):
    
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