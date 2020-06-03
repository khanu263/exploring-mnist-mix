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
import torch.optim as optim
import matplotlib.pyplot as plt

# Imports - custom
import models
import opt

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--create", nargs = "*")
parser.add_argument("--load")
parser.add_argument("--labels")
parser.add_argument("--train", nargs = 5)
parser.add_argument("--test", nargs = "+")
parser.add_argument("--log")
parser.add_argument("--save")
parser.add_argument("--gpu", default = False, action = "store_true")
args = parser.parse_args()

# Check create/load specifications
if args.create and args.load:
    sys.exit("You cannot create and load a model in the same run.")
elif args.create:
    MODEL_TYPE = args.create[0]
    MODEL_PARAMS = list(map(int, args.create[1:]))
elif args.load:
    LOAD_PATH = args.load

# Check label specification
if not args.labels or args.labels not in ("specific", "agnostic"):
    sys.exit("Must specify labels as 'specific' or 'agnostic'.")
LABEL_TYPE = args.labels
NUM_CLASSES = 100 if LABEL_TYPE == "specific" else 10

# Check that testing is defined if training is
if args.train and not args.test:
    sys.exit("If training is defined, testing must be as well.")

# Check training specification
if args.train:
    TRAIN_FILE = args.train[0]
    EPOCHS = int(args.train[1])
    BATCH_SIZE = int(args.train[2])
    LEARNING_RATE = float(args.train[3])
    MOMENTUM = float(args.train[4])
    if EPOCHS < 1 or BATCH_SIZE < 1:
        sys.exit("Epochs and batch size must be at least 1.")
    if LEARNING_RATE < 0 or MOMENTUM < 0:
        sys.exit("Learning rate and momentum must be at least 0.")

# Check testing specification
if args.test:
    TEST_FILE = args.test[0]
    try:
        SAVE_TEST = args.test[1]
    except:
        SAVE_TEST = None

# Get the rest of the specifications
if args.log:
    LOG_PATH = args.log
if args.save:
    SAVE_PATH = args.save

# Get the device to use
DEVICE = torch.device("cuda:0" if (args.gpu and torch.cuda.is_available()) else "cpu")
print("\nSelected device: {}".format(DEVICE))

# Create or load the specified model
if args.load:
    model = torch.load(LOAD_PATH)
elif MODEL_TYPE == "feedforward":
    model = models.FeedForward(MODEL_PARAMS, NUM_CLASSES)
else:
    model = models.ResNet(MODEL_PARAMS, NUM_CLASSES)
model.to(DEVICE)
print("Created/loaded model and moved to specified device.")

# Infer if using ResNet or feedforward
if model.__class__.__name__ == "ResNet":
    USE_RESNET = True
else:
    USE_RESNET = False
print("Using ResNet? {}".format(USE_RESNET))

# Load data and labels
data = torch.from_numpy(np.load("data/data.npy"))
labels = torch.from_numpy(np.load("data/ls_labels.npy") if LABEL_TYPE == "specific" else np.load("data/la_labels.npy"))
print("Loaded data and labels.")

# Load training split if specified
if args.train:
    train_split = torch.from_numpy(np.loadtxt(TRAIN_FILE, dtype = int))
    train_set = torch.utils.data.TensorDataset(train_split)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)
    print("Loaded training split.")

# Load testing and validation splits if specified
if args.test:

    # Testing split is guaranteed
    test_split = torch.from_numpy(np.loadtxt(TEST_FILE, dtype = int))
    test_set = torch.utils.data.TensorDataset(test_split)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 64)
    print("Loaded testing split.")

    # Load validation split only if we are training
    if args.train:
        val_split = test_split[::10]
        val_set = torch.utils.data.TensorDataset(val_split)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size = BATCH_SIZE)
        print("Created validation split.")

# Perform training if specified
if args.train:

    # Initialize optimizer loss tracking lists
    optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)
    train_losses = []
    val_losses = []
    accuracies = []
    print("\nTraining model for {} epochs with batch size {}, learning rate {}, and momentum {}."
          .format(EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM))

    # Train the model for the specified number of epochs
    for e in range(EPOCHS):

        print("\nStarting epoch {}.\n".format(e))

        # Perform a training and validation
        train_losses.append(opt.train(model, data, labels, train_loader, optimizer, DEVICE, USE_RESNET))
        val_losses.append(opt.validate(model, data, labels, val_loader, DEVICE, USE_RESNET))

        # Test the model
        total, correct, confusion_matrix, softmax_matrix = opt.test(model, data, labels, test_loader, DEVICE, NUM_CLASSES, USE_RESNET)
        accuracies.append(correct / total)

        # Print progress
        print("\nFinished epoch {}. Training loss: {:.2f}. Validation loss: {:.2f}. Accuracy: {:.2f}%."
            .format(e, train_losses[-1], val_losses[-1], accuracies[-1] * 100))

    # If specified, save the log
    if args.log:

        # Generate lines
        train_lines = "\t".join(map(str, train_losses))
        val_lines = "\t".join(map(str, val_losses))
        acc_lines = "\t".join(map(str, accuracies))

        # Write file
        with open(LOG_PATH, "w") as f:
            f.write("\n".join([train_lines, val_lines, acc_lines]))

        # Report success
        print("\nWrote log of losses and accuracies to {}.".format(LOG_PATH))

# Perform a final round of testing if specified
if args.test:

    # Do the testing and report results
    total, correct, confusion_matrix, softmax_matrix = opt.test(model, data, labels, test_loader, DEVICE, NUM_CLASSES, USE_RESNET)
    print("\nPerformed testing. Final accuracy: {:.2f}%. ({} / {})".format((correct / total) * 100, correct, total))

    # If requested, save to file with confusion matrix
    if SAVE_TEST:

        # Convert confusion matrix to NumPy array and save
        save_conf = confusion_matrix.to("cpu").numpy()
        np.savetxt(SAVE_TEST, save_conf, fmt = "%s")

        # Append the accuracy to the end
        with open(SAVE_TEST, "a") as f:
            f.write("\n{}".format((correct / total) * 100))

        # Report success
        print("\nWrote confusion matrix and accuracy to {}.".format(SAVE_TEST))

# Save the model if requested
if args.save:
    model.to("cpu")
    torch.save(model, SAVE_PATH)
    print("\nSaved model to {}.".format(SAVE_PATH))

# Final newline
print("")