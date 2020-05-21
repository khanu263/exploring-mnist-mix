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

# Load raw data
x_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
print("Loaded raw data.")

# Create full dataset and scale
data = np.concatenate((x_train, x_test))
data = data / 255
print("Created and scaled full dataset.")

# Create full sets of labels
ls_labels = np.concatenate((y_train, y_test)).astype(np.int32)
la_labels = np.mod(ls_labels, 10).astype(np.int32)
print("Created language-specific and language-agnostic labels.")

# Initialize train and test splits
cutoff = len(x_train)
end = len(ls_labels)
all_train = []
all_test = []
l_train = []
l_test = []
for i in range(10):
    l_train.append([])
    l_test.append([])

# Go through data and assign indices
for i in range(cutoff):
    all_train.append(i)
    l_train[ls_labels[i] // 10].append(i)
for i in range(cutoff, end):
    all_test.append(i)
    l_test[ls_labels[i] // 10].append(i)
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