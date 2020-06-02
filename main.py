
# Imports
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from opt import test, training, train, validation
from feedforward import FeedForward
def main():
# Load raw data
    dataset = 'Arabic_train_test.npz'
    data = np.load(dataset)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_train = X_train.type(torch.FloatTensor)
    y_train = y_train.type(torch.FloatTensor)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)
    X_test = X_test.type(torch.FloatTensor)
    y_test = y_test.type(torch.FloatTensor)
    X_train = X_train / 255
    X_test = X_test / 255
    # percentage of training set to use as validation
    valid_size = 0.2
    batch_size = 10

    # Pytorch train sets
    train_data = torch.utils.data.TensorDataset(X_test, y_test)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining validation batches
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)
    net = FeedForward([100,100])
    if (torch.cuda.is_available()):
        net = net.cuda()
    accuracy, confusion, softmax = test(X_test, y_test, net)
    print(accuracy)
    for i in range(20):
        training(X_train[train_idx], y_train[train_idx], net)
        accuracy, confusion, softmax = test(X_test, y_test, net)
        print(accuracy)
    
    validation(valid_loader, net, 2)

if __name__ == '__main__':
    main()

