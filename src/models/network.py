import torch.nn as nn
import numpy as np


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(1, 1000)  # Input layer to hidden layer
        self.layer2 = nn.Linear(1000, 1000)   # Hidden layer to another hidden layer
        self.layer3 = nn.Linear(1000, 5)
        self.output_layer = nn.Linear(5, 1)  # Hidden layer to output layer
        self.activation = nn.ReLU()  # Activation function

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.output_layer(x)
        return x