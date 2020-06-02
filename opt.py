# opt.py
# CS 445/545 -- Spring 2020

# Training and validation / testing functions.

# Imports
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#################################################
# TRAINING
#################################################

# Train the given model on the given data with the given parameters.
def train(model, data, labels, loader, lr, momentum, device):

    # Define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum)

    # Initialize loss for epoch
    total_loss = 0

    # Go through each batch
    for batch in loader:

        # Get the data and labels
        data_batch = data[batch].to(device)
        label_batch = labels[batch].to(device)

        # Zero the optimizer gradients
        optimizer.zero_grad()

        # Pass the batch through the network
        output = F.softmax(model(data_batch), dim = 1)

        # Create the target based on the label batch
        target = torch.zeros_like(output)
        for i in range(len(target)):
            target[i][label_batch[i]] = 1

        # Calculate loss and run optimization step
        loss = criterion(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    # Return average loss
    return total_loss / len(loader)

#################################################
# VALIDATION
#################################################

def validation(valid_loader, model, epoch):

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
    return valid_loss

#################################################
# TESTING
#################################################

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