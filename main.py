# main.py
# CS 445/545 -- Spring 2020

# Main coordination script for experiments.

# Imports - libraries
import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Imports - custom
import models
import opt

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--feedforward", type = int, nargs = "*")
parser.add_argument("--resnet", type = int, nargs = "*")
parser.add_argument("--epochs", type = int)
parser.add_argument("--batch-size", type = int)
parser.add_argument("--train")
parser.add_argument("--test")
parser.add_argument("--labels")
parser.add_argument("--learning-rate", type = float)
parser.add_argument("--momentum", type = float)
parser.add_argument("--log")
parser.add_argument("--save")
parser.add_argument("--gpu", default = False, action = "store_true")
args = parser.parse_args()

# Get the device to use
device = torch.device("cuda:0" if (args.gpu and torch.cuda.is_available()) else "cpu")
print("\nSelected device: {}".format(device))

# Load data, labels, and splits
data = np.load("data/data.npy")
labels = np.load("data/la_labels.npy") if args.labels == "agnostic" else np.load("data/ls_labels.npy")
train_split = np.loadtxt(args.train, dtype = int)
test_split = np.loadtxt(args.test, dtype = int)
val_split = test_split[::10]
print("Loaded NumPy data.")

# Convert to tensor
data = torch.from_numpy(data)
labels = torch.from_numpy(labels)
train_split = torch.from_numpy(train_split)
test_split = torch.from_numpy(test_split)
val_split = torch.from_numpy(val_split)
print("Converted data to tensor format.")

# Define data sets and data loaders
train_set = torch.utils.data.TensorDataset(train_split)
test_set = torch.utils.data.TensorDataset(test_split)
val_set = torch.utils.data.TensorDataset(val_split)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = args.batch_size)
val_loader = torch.utils.data.DataLoader(val_set, batch_size = args.batch_size)
print("Created data loaders.")

# Get the correct number of classes
if args.labels == "agnostic":
    num_classes = 10
    print("Selected agostic labels.")
elif args.labels == "specific":
    num_classes = 100
    print("Selected specific labels.")
else:
    sys.exit("Labels must be 'agnostic' or 'specific'.")

# Create the specified model
if args.feedforward:
    model = models.FeedForward(args.feedforward, num_classes)
    print("Created feedforward model with parameters {}".format(args.feedforward))
elif args.resnet:
    model = models.ResNet(models.ResNetBasicBlock, args.resnet, num_classes)
    print("Created ResNet model with parameters {}".format(args.resnet))
else:
    sys.exit("No model specified. Exiting.")

# Move model to specified device and initialize loss tracking lists
model.to(device)
train_losses = []
val_losses = []
accuracies = []
print("Moved model to device and initialized loss lists.")

# Train the model for the specified number of epochs
for e in range(args.epochs):

    print("\nStarting epoch {}.\n".format(e))

    # Perform a training and validation
    train_losses.append(opt.train(model, data, labels, train_loader, args.learning_rate, args.momentum, device))
    val_losses.append(opt.validate(model, data, labels, val_loader, device))

    # Test the model
    total, correct, confusion_matrix, softmax_matrix = opt.test(model, data, labels, test_loader, device, num_classes)
    accuracies.append(correct / total)

    # Print progress
    print("\nFinished epoch {}. Training loss: {:.2f}. Validation loss: {:.2f}. Accuracy: {:.2f}%."
          .format(e, train_losses[-1], val_losses[-1], accuracies[-1] * 100))