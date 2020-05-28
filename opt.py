import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
def training(data, label, model):
    batch_size = 16
    epochs = 10
    learning_rate = 1e-2
    momentum_value = 1e-1
    runs_per_epoch = int(data.shape[0] / batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_value)
    for epoch in range(epochs):
        permutation = np.random.permutation(data.shape[0])
        data = data[permutation]
        label = label[permutation]
        for i in range(runs_per_epoch):
            train(data[i*batch_size: (i+1)*batch_size - 1,:,:], label[i*batch_size: (i+1)*batch_size - 1], model, optimizer)
        if (batch_size * runs_per_epoch == data.shape[0]): # perfectly divided batches
            continue
        train(data[i * runs_per_epoch:,:,:], label[i * runs_per_epoch:], model, optimizer)


def train(data, label, model, optimizer):
    optimizer.zero_grad()
    output = model(data)
    target = torch.zeros_like(output)
    for p in range(target.shape[0]):
        target[p, int(label[p])] = 1
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()