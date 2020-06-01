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
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = momentum_value)

    # percentage of training set to use as validation
    valid_size = 0.2

    # Pytorch train sets
    train_data = torch.utils.data.TensorDataset(data, label)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining validation batches
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)

    for epoch in range(epochs):
        permutation = np.random.permutation(data.shape[0])
        data = data[permutation]
        label = label[permutation]
        for i in range(runs_per_epoch):
            train(data[i*batch_size: (i+1)*batch_size - 1,:,:], label[i*batch_size: (i+1)*batch_size - 1], model, optimizer)
        if (batch_size * runs_per_epoch == data.shape[0]): # perfectly divided batches
            continue
        train(data[i * runs_per_epoch:,:,:], label[i * runs_per_epoch:], model, optimizer)
        validation(valid_loader, epoch)

def train(data, label, model, optimizer):
    if (torch.cuda.is_available()):
        data = data.cuda()
        label = label.cuda()
    output = model(data)
    target = torch.zeros_like(output)
    for p in range(target.shape[0]):
        target[p, int(label[p])] = 1
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def validation(valid_loader, epoch):

    # Loss function
    criterion = nn.MSELoss()

    # keep track of validation loss
    valid_loss = 0.0

    # validate the model
    # tell the model that it is evaluation mode
    # model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss
        valid_loss += loss.item() * data.size(0)

    # calculate average losses
    valid_loss = valid_loss / len(valid_loader.sampler)

    # print validation statistics
    print('Epoch: {} \tValidation Loss: {:.6f}'.format(
        epoch, valid_loss))

def test(data, label, model):
    if (torch.cuda.is_available()):
        data = data.cuda()
        label = label.cuda()
    output = model(data)
    accuracy = 0
    confusion_matrix = torch.zeros((output.shape[1], output.shape[1]))
    softmax_matrix = torch.zeros((output.shape[1], output.shape[1]))
    if (torch.cuda.is_available()):
        confusion_matrix = confusion_matrix.cuda()
        softmax_matrix = softmax_matrix.cuda()
    for p in range(label.shape[0]):
        prediction = torch.argmax(output[p,:])
        softmax = F.softmax(output[p,:])
        confusion_matrix[int(label[p]), prediction] += 1
        softmax_matrix[int(label[p]), :] += softmax[:]
        if (prediction == label[p]):
            accuracy += 1
    return accuracy / output.shape[0], confusion_matrix, softmax_matrix