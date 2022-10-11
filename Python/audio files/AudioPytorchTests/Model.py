'''
Script containing the neural network and that can launch train or test
'''

import numpy as np
import matplotlib

import torch
from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor # takes an image and converts it in a normalized tensor

import matplotlib.pyplot as plt


# ================================ Model as a class ================================================ #
# Needs a constructor and a forward method
class FeedForwardNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten() # First Layer 
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        # indicates pytorch how to manipulate data
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data) # outputs
        predictions = self.softmax(logits)
        return predictions

# TODO: Create a class for a model that supports audio files instead of 28*28 pixels with a 0-255 greyscale range
class CNNNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        # input are greyscale mel spectrograms
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128 ,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        # 
        self.linear = nn.Linear(128 * 5 * 4, 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions
        

# ================================ Defining metaparameters ========================================= #
sampleRate = 16000
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

# ================================ Accessing data directories ====================================== #


# ================================ Downloading examples data  ====================================== #
def download_mnist_datasets():
    # Test d'abord sur le set de 10 chiffres
    train_data = datasets.MNIST(
        root = "data",
        download=True,
        train=True,
        transform=ToTensor()
    )

    validation_data = datasets.MNIST(
        root = "data",
        download=True,
        train=False,
        transform=ToTensor()
    )

    return train_data, validation_data

def MNIST_demo_execute():
    train_data, _ = download_mnist_datasets()
    print('downloading mnist datasets')

    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # build model
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f'using {device} device')
    feed_forward_net = FeedForwardNet().to(device)

    # instanciate loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(), lr=LEARNING_RATE)

    # train model
    train(feed_forward_net, train_data_loader, loss_fn, optimiser, device, EPOCHS)
    
    # Store the model once it is trained
    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth") # state_dict =
    print("Models trained and stored at feedforwardnet.pth")

# ================================ Méthodes essentielles pour train ====================================== #
def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    # lower level
    # loop through all samples
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagate loss and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Loss: {loss.item()}")
    

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    # Higher level, goes through all epochs
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("-----------------------------")
    print("Training is done!")
    

# ================================  ====================================== #

if __name__ == "__main__":
    cnn = CNNNetwork()
    # tuple is (number of input channel of network, number of mels, and time axis)
    summary(cnn, (1, 64, 44))
