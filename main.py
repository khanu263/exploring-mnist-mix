
# Imports
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from opt import test, training, train
from feedforward import FeedForward
def main():
# Load raw data
    dataset = 'Arabic_train_test.npz'
    data = np.load(dataset)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    X_train = X_train / 255
    X_test = X_test / 255
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_train = X_train.type(torch.FloatTensor)
    y_train = y_train.type(torch.FloatTensor)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)
    X_test = X_test.type(torch.FloatTensor)
    y_test = y_test.type(torch.FloatTensor)
    net = FeedForward()
    if (torch.cuda.is_available()):
        net = net.cuda()
    print(test(X_test, y_test, net))
    for i in range(500):
        training(X_train, y_train, net)
        print(test(X_test, y_test, net))


if __name__ == '__main__':
    main()

