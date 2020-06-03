# opt.py
# CS 445/545 -- Spring 2020

# Training and validation / testing functions.

# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#################################################
# TRAINING
#################################################

# Train the given model on the given data.
def train(model: nn.Module, data: torch.Tensor, labels: torch.Tensor, loader: torch.utils.data.dataloader.DataLoader,
          optimizer: optim.Optimizer, device: torch.device, resnet: bool) -> float:

    # Switch to train
    model.train()

    # Define criterion and initialize loss
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    # Go through each batch
    for i, batch in enumerate(loader):

        # Get the data and labels
        data_batch = data[batch[0].long()].to(device)
        label_batch = labels[batch[0].long()].clone().detach().long().to(device)

        # Reshape batch if using a ResNet
        if resnet:
            data_batch = data_batch.reshape(len(data_batch), 1, 28, 28)

        # Zero the optimizer gradients
        optimizer.zero_grad()

        # Pass the batch through the network
        output = F.softmax(model(data_batch), dim = 1)

        # Calculate loss and run optimization step
        loss = criterion(output, label_batch)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        # Print progress
        if i % 2500 == 0:
            print("\tFinished batch {} ({:.1f}% complete).".format(i, (i / len(loader)) * 100))

    # Return average loss
    return total_loss / len(loader)

#################################################
# VALIDATION
#################################################

# Run validation on the given model with the given data.
def validate(model: nn.Module, data: torch.Tensor, labels: torch.Tensor, loader: torch.utils.data.dataloader.DataLoader,
             device: torch.device, resnet: bool) -> float:

    # Switch to eval
    model.eval()

    # Initialize loss and define criterion
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    # Turn off gradient tracking
    with torch.no_grad():

        # Go through each batch
        for batch in loader:

            # Get the data and labels
            data_batch = data[batch[0].long()].to(device)
            label_batch = labels[batch[0].long()].clone().detach().long().to(device)

            # Reshape batch if using a ResNet
            if resnet:
                data_batch = data_batch.reshape(len(data_batch), 1, 28, 28)

            # Pass the batch through the network
            output = F.softmax(model(data_batch), dim = 1)

            # Calculate loss
            loss = criterion(output, label_batch)
            total_loss += loss.item()

    # Return average loss
    return total_loss / len(loader)

#################################################
# TESTING
#################################################

# Test the model (accuracy / confusion) on the given model with the given data.
def test(model: nn.Module, data: torch.Tensor, labels: torch.Tensor, loader: torch.utils.data.dataloader.DataLoader,
         device: torch.device, num_classes: int, resnet: bool) -> (int, int, torch.Tensor, torch.Tensor):

    # Switch to eval mode
    model.eval()

    # Initialize metrics
    total = 0
    correct = 0
    confusion_matrix = torch.zeros((num_classes, num_classes))
    softmax_matrix = torch.zeros((num_classes, num_classes))

    # Turn off gradient tracking
    with torch.no_grad():

        # Go through each batch
        for batch in loader:

            # Get the data and labels
            data_batch = data[batch[0].long()].to(device)
            label_batch = labels[batch[0].long()].to(device)

            # Reshape batch if using a ResNet
            if resnet:
                data_batch = data_batch.reshape(len(data_batch), 1, 28, 28)

            # Pass the batch through the network and get the prediction
            output = F.softmax(model(data_batch), dim = 1)

            # Move to CPU for caluclations
            output = output.to("cpu")
            label_batch = label_batch.to("cpu")

            # Iterate through each example
            for i in range(len(data_batch)):

                # Get the prediction and the actual label
                pred = int(torch.argmax(output[i]))
                label = int(label_batch[i])

                # Increment accuracy counts
                total += 1
                if pred == label:
                    correct += 1

                # Increment matrices
                confusion_matrix[label, pred] += 1
                softmax_matrix[label, :] += output[i]

    # Move softmax matrix back to CPU and return metrics
    return total, correct, confusion_matrix, softmax_matrix