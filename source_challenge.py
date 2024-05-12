"""
EECS 445 - Introduction to Machine Learning
Winter 2024 - Project 2

Source CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.source import Source
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Source_Challenge(nn.Module):
    def __init__(self):
        """
        Define the architecture, i.e. what layers our network contains.
        At the end of __init__() we call init_weights() to initialize all model parameters (weights and biases)
        in all layers to desired distributions.
        """
        super().__init__()

        # TODO: define each layer
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size,
        # stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        ## TODO: define your model architecture
        self.conv1 = nn.Conv2d(3, 16, 5, 2, 2)
        self.conv2 = nn.Conv2d(16, 64, 5, 2, 2)
        self.conv3 = nn.Conv2d(64, 8, 5, 2, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32, 8)

        self.init_weights()

    def init_weights(self):
        """Initialize all model parameters (weights and biases) in all layers to desired distributions"""

        torch.manual_seed(42)
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        ## TODO: initialize the parameters for [self.fc1]
        f_in = self.fc1.in_features
        nn.init.normal_(self.fc1.weight, 0.0, 1 / sqrt(f_in))
        nn.init.constant_(self.fc1.bias, 0.0)

    def forward(self, x):
        """
        This function defines the forward propagation for a batch of input examples, by
        successively passing output of the previous layer as the input into the next layer (after applying
        activation functions), and returning the final output as a torch.Tensor object.

        You may optionally use the x.shape variables below to resize/view the size of
        the input matrix at different points of the forward pass.
        """
        N, C, H, W = x.shape
        ## TODO: implement forward pass for your network
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.max_pool2d(x, 2)
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = F.max_pool2d(x, 2)
        #print(x.shape)
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        

        return x

        ## TODO: forward pass
