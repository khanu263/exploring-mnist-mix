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
# import opt

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

# Load data, labels, and splits
data = np.load("data/data.npy")
labels = np.load("data/la_labels.npy") if args.labels == "agnostic" else np.load("data/ls_labels.npy")
train_split = np.loadtxt(args.train, dtype = int)
test_split = np.loadtxt(args.test, dtype = int)
val_split = test_split[::10]

# Convert to tensor
data = torch.from_numpy(data)
labels = torch.from_numpy(labels)
train_split = torch.from_numpy(train_split)
test_split = torch.from_numpy(test_split)
val_split = torch.from_numpy(val_split)

# Define data sets and data loaders
train_set = torch.utils.data.TensorDataset(train_split)
test_set = torch.utils.data.TensorDataset(test_split)
val_set = torch.utils.data.TensorDataset(val_split)
train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = args.batch_size)
val_loader = torch.utils.data.DataLoader(val_set, batch_size = args.batch_size)

# def main():
# # Load raw data
#     dataset = 'Arabic_train_test.npz'
#     data = np.load(dataset)
#     X_train = data['X_train']
#     y_train = data['y_train']
#     X_test = data['X_test']
#     y_test = data['y_test']
#     X_train = torch.from_numpy(X_train)
#     y_train = torch.from_numpy(y_train)
#     X_train = X_train.type(torch.FloatTensor)
#     y_train = y_train.type(torch.FloatTensor)
#     X_test = torch.from_numpy(X_test)
#     y_test = torch.from_numpy(y_test)
#     X_test = X_test.type(torch.FloatTensor)
#     y_test = y_test.type(torch.FloatTensor)
#     X_train = X_train / 255
#     X_test = X_test / 255
#     # percentage of training set to use as validation
#     valid_size = 0.2
#     batch_size = 10

#     # Pytorch train sets
#     train_data = torch.utils.data.TensorDataset(X_test, y_test)

#     # obtain training indices that will be used for validation
#     num_train = len(train_data)
#     indices = list(range(num_train))
#     np.random.shuffle(indices)
#     split = int(np.floor(valid_size * num_train))
#     train_idx, valid_idx = indices[split:], indices[:split]

#     # define samplers for obtaining validation batches
#     valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

#     # prepare data loaders (combine dataset and sampler)
#     valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)
#     net = FeedForward([100,100])
#     if (torch.cuda.is_available()):
#         net = net.cuda()
#     accuracy, confusion, softmax = test(X_test, y_test, net)
#     print(accuracy)
#     for i in range(100):
#         training(X_train[train_idx], y_train[train_idx], net)
#         accuracy, confusion, softmax = test(X_test, y_test, net)
#         print(accuracy)
#     val = validation(valid_loader, net, 2)
#     net2 = FeedForward([100,100,100])
#     if (torch.cuda.is_available()):
#         net2 = net2.cuda()
#     accuracy, confusion, softmax = test(X_test, y_test, net)
#     print(accuracy)
#     for i in range(100):
#         training(X_train[train_idx], y_train[train_idx], net2)
#         accuracy, confusion, softmax = test(X_test, y_test, net2)
#         print(accuracy)
#     val2 = validation(valid_loader, net2, 2)
#     if (val2 < val):
#         net = net2
#         val = val2
#     net3 = FeedForward([100,100,100])
#     if (torch.cuda.is_available()):
#         net3 = net3.cuda()
#     accuracy, confusion, softmax = test(X_test, y_test, net)
#     print(accuracy)
#     for i in range(100):
#         training(X_train[train_idx], y_train[train_idx], net3)
#         accuracy, confusion, softmax = test(X_test, y_test, net3)
#         print(accuracy)
#     val3 = validation(valid_loader, net3, 2)

#     if (val3 < val):
#         net = net3
#     torch.save(net, "feedforward.pt")
    

# if __name__ == '__main__':
#     main()

