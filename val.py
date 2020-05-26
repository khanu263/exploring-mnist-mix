# val.py
# CS 445/545 -- Spring 2020

# Validation function for network training.

# Imports
import torch
import numpy as np

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2

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

# Loss function
criterion = nn.CrossEntropyLoss()

# number of epochs to train the model
n_epochs = 30

for epoch in range(1, n_epochs + 1):

    # keep track of validation loss
    valid_loss = 0.0

    # validate the model
    # tell the model that it is evaluation mode
    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
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


