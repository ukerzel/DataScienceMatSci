##
## This is the non-interactive version of the MNIST example using PyTorch
## It is intended to show how to run the training from a script, e.g. on a cluster.
##

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchvision as tv

import matplotlib.pyplot as plt
import numpy as np

## ##########################################
## Train / Test loops
## ##########################################

#
# Training loop
#
def train_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    running_loss = 0.0

    # put the model into training mode
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss functions // change loss
        loss_item = loss.item()
        current = batch * len(X)

        running_loss += loss_item
        if batch % 100 == 0:
            #loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss_item:>7f}  [{current:>5d}/{size:>5d}]")

    return running_loss/size

#
# Test Loop
#
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n") 
    return correct


## ##########################################
## Network definition
## ##########################################

# Define  a simple model
class NeuralNetwork(nn.Module):
   def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

   def forward(self, x):
       x = self.flatten(x)
       #logits = self.linear_relu_stack(x)
       # return logits
       x = self.fc1(x)
       x = F.relu(x)
       x = self.fc2(x)
       x = F.relu(x)
       x = self.fc3(x)
       return x

## ##########################################
## Main program
## ##########################################

if __name__ == '__main__':

    #
    # general setups
    #
    batch_size = 64

    # Download data from open datasets.
    training_data = tv.datasets.MNIST(
        root="data/mnist",
        train=True,
        download=True,
        transform=tv.transforms.ToTensor(),
    )

    test_data = tv.datasets.MNIST(
        root="data/mnist",
        train=False,
        download=True,
        transform=tv.transforms.ToTensor(),
    )
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")


    # Instantiate a model
    model = NeuralNetwork().to(device)
    print(model)


    # Optimizer and Loss Function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ##
    ## The actual training/evaluation
    ##
    epochs = 5
    loss_values = []
    accuracy_values = []

    for t in range(epochs):

        print(f"Epoch {t+1}\n-------------------------------")
        loss = train_epoch(train_dataloader, model, loss_fn, optimizer)
        loss_values.append(loss)

        accuracy = test(test_dataloader, model, loss_fn)
        accuracy_values.append(accuracy)
    
    # save the model for later use
    torch.save(model.state_dict(), "mnist.pt")
    print("Done!")

