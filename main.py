#  y = sin(3.5x1 + 1)*cos(5.5x2)
# constrain x1, x2 in [-1, 1]

import numpy
import matplotlib.pyplot as plt
import random
import math
import torch
import torch.nn as nn

class Net(torch.nn.Module):
    # initialise one hidden layer and one output layer
    def init(self, n_feature, n_hidden, n_output):
        # call pytorch superclass code to initialise nn
        super(Net, self).init()
        # hidden layer
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)  # output layer

    # connect up the layers: the input passes through the hidden, then the sigmoid, then the output layer
    # x is the data8
    def forward(self, x):
        # activation function for hidden layer is relu
        # data passes through hidden and relu
        x = F.relu(self.hidden(x))
        # the result needs to pass through output layer next
        x = self.out(x)
        return x

net = Net(n_feature=9, n_hidden=2, n_output=2)  # define the network

#optional printouts about network
print("printing net")
print(net)  # net architecture




