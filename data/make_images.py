# make_images.py
# CS 445/545 -- Spring 2020

# Use this script with the processed MNIST-MIX data to
# save all examples as images to an 'images' directory.

# Usage: place this in the same directory as data.npy and
#        ls_labels.npy, then run.

# Imports
import os
import cv2
import numpy as np

# Make directories if they don't exist
if not os.path.exists("images"):
    os.makedirs("images")
for i in range(100):
    if not os.path.exists("images/{:0>2d}".format(i)):
        os.makedirs("images/{:0>2d}".format(i))
print("Made directories.")

# Load data
data = np.load("data.npy")
ls_labels = np.load("ls_labels.npy")
print("Loaded data.")

# Rescale the data back to 0-255
data = data * 255
print("Rescaled data.")

# Save images
for i in range(len(data)):
    cv2.imwrite("images/{:0>2d}/{}.png".format(ls_labels[i], i), data[i])
    if i % 50000 == 0:
        print("Wrote image {}.".format(i))
print("Finished saving images.")