#  y = sin(3.5x1 + 1)*cos(5.5x2)
# constrain x1, x2 in [-1, 1]

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

torch.manual_seed(1)  # reproducible experiments


# by fixing the seed you will remove randomness
# create the NN
class Net(nn.Module):
    # initialise one hidden layer and one output layer
    def __init__(self, n_feature, n_hidden, n_output):
        # call pytorch superclass code to initialise nn
        super(Net, self).__init__()
        # hidden layer
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)  # output layer

    # connect up the layers: the input passes through the hidden, then the sigmoid, then the output layer
    # x is the data
    def forward(self, x):
        # activation function for hidden layer is sigmoid
        # data passes through hidden and sigmoid
        x = nn.functional.sigmoid(self.hidden(x))
        # the result needs to pass through output layer next
        x = self.out(x)
        return x


# Returns z vale for given x and y
def function(x1, x2):
    return np.sin(3.5 * x1 + 1) * np.cos(5.5 * x2)


# creates a 3d plot of the function
def visualfunction(x1arr, x2arr, title):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x1 = np.linspace(-2.1, 2.1, 100)
    x2 = np.linspace(-1.1, 1.1, 100)
    x1, x2 = np.meshgrid(x1, x2)
    y = function(x1, x2)

    # create the surface
    surfaceplot = ax.plot_surface(x1, x2, y, rstride=1, cstride=1, cmap=cm.viridis, alpha=0.8, linewidth=0,
                                  antialiased=False)
    ax.view_init(45, 45)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surfaceplot, shrink=0.5)
    plt.show()


# plots a 3D scatter plot
def scatter3D(x1arr, x2arr, yarr, title):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x1arr, x2arr, yarr, c=yarr, marker='o')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('z')
    ax.set_title(title)
    plt.show()


def main():
    # creating dataset in order to train and test the model
    # creating 1100 random values for x1 and x2 between -1 and 1
    x1arr = np.random.uniform(-1, 1, 1100)
    x2arr = np.random.uniform(-1, 1, 1100)
    # creating a list of the function values
    yarr = function(x1arr, x2arr)

    # create a 3d plot of the function
    visualfunction(x1arr, x2arr, 'All Data')

    # creating a list of the training data
    x1arrtrain = torch.as_tensor(x1arr[:1000], dtype=torch.double)
    x2arrtrain = torch.as_tensor(x2arr[:1000], dtype=torch.double)
    yarrtrain = torch.as_tensor(yarr[:1000], dtype=torch.double)

    # plot the training data as a 3D scatter plot
    scatter3D(x1arrtrain, x2arrtrain, yarrtrain, title='Training Data')

    # creating a list of the testing data
    x1arrtest = torch.as_tensor(x1arr[1000:1100], dtype=torch.double)
    x2arrtest = torch.as_tensor(x2arr[1000:1100], dtype=torch.double)
    yarrtest = torch.as_tensor(yarr[1000:1100], dtype=torch.double)

    # plot testing data as a 3D scatter plot
    scatter3D(x1arrtest, x2arrtest, yarrtest, title='Testing Data')

    net = Net(n_feature=6, n_hidden=2, n_output=2)  # define the network

    # optional printouts about network
    print("printing net")
    print(net)  # net architecture


if __name__ == "__main__":
    main()
