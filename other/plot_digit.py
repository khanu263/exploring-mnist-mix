# models.py
# CS 445/545 -- Spring 2020

# Function to plot a single digit.

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Plot the given digit
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