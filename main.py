#  y = sin(3.5x1 + 1)*cos(5.5x2)
# constrain x1, x2 in [-1, 1]

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random as r
from sympy.combinatorics.graycode import gray_to_bin, bin_to_gray

torch.manual_seed(1)  # reproducible experiments

maxnum = 2 ** 30
minRange = -20
maxRange = 20


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


# method for adjusting three random weights of the network in the input layer
def adjustweights(weights):
    changedweights = []

    change1 = r.randint(0, 18)
    change2 = r.randint(0, 18)
    change3 = r.randint(0, 18)

    if change1 == change2:
        change2 += 1
    elif change2 == change3 or change1 == change3:
        change3 += 1

    multiplier1 = r.uniform(0, 1)
    multiplier2 = r.uniform(0, 1)
    multiplier3 = r.uniform(0, 1)

    weights[change1] = weights[change1] * multiplier1
    weights[change2] = weights[change2] * multiplier2
    weights[change3] = weights[change3] * multiplier3

    return weights


# method for reshaping a list into a np array the network will accept
def reformList(list, rows, cols):
    list = np.reshape(list, (rows, cols))
    return list


# extract all the weights of the net and puts them into a list
def weightsOutofNetwork(net):
    weights = []
    weightlist = []

    for param in net.parameters():
        weights.append(param.data.numpy().flatten())

    # flatteing the weights into a 2d array with tensor values assigned to each weight
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            weightlist.append(weights[i][j])

    return weightlist


# takes as in a list of all weights of the network and uses them to set the weights of the network
def weightsIntoNetwork(weights, net):
    net.hidden.weight = torch.nn.Parameter(torch.from_numpy(reformList(weights[:12], 6, 2)))
    net.hidden.bias = torch.nn.Parameter(torch.from_numpy(reformList(weights[12:18], 6, 1)))
    net.hidden2.weight = torch.nn.Parameter(torch.from_numpy(reformList(weights[18:54], 6, 6)))
    net.hidden2.bias = torch.nn.Parameter(torch.from_numpy(reformList(weights[54:60], 6, 1)))
    net.out.weight = torch.nn.Parameter(torch.from_numpy(reformList(weights[60:66], 1, 6)))
    net.out.bias = torch.nn.Parameter(torch.from_numpy(reformList(weights[66:67], 1, 1)))
    return net


# Convert chromosome to real number
# input: list binary 1,0 of length numOfBits representing number using gray  coding
# output: real value of chromosome
def chrom2real(c, minRange, maxRange):
    indasstring = ''.join(map(str, c))

    degray = gray_to_bin(indasstring)
    numasint = int(degray, 2)  # convert to int from base 2 list
    numinrange = minRange + (maxRange - minRange) * numasint / maxnum
    return numinrange


# converts a list of weights into a list with the weights as chromosomes
# range of weights is between -20 and 20
# 6 bits for integer part of weight and 24 bits for fractional part of weight
# add 20 to all numbers to make sure they are positive
# round to 7 dp to prevent overflow
# convert to binary
# convert to gray
def real2Chrom(weights):
    chroms = []

    for i in range(len(weights)):


        # ensures the weights are in the range of -20 and 20
        if weights[i] > maxRange:
            weights[i] = maxRange
        elif weights[i] < minRange:
            weights[i] = minRange

        # round to 7 dp to prevent overflow and value between 0 and 1
        numPrepped = round((weights[i] + maxRange)/40, 7)

        # split float into two parts, one for the integer part and one for the decimal part
        integer, decimal = str(numPrepped).split('.')

        print(str(numPrepped).split('.'))

        # convert the integer part to binary
        integer = bin(int(integer))[2:]

        decimal = bin(int(decimal))[2:]

        # pad the binary with 0's to make it the correct length
        integer = integer.zfill(6-len(integer))
        decimal = decimal.zfill(24-len(decimal))

        # combine the two parts into one string
        numInBits = integer + decimal  # convert value to base 2
        gray = bin_to_gray(numInBits)  # convert to gray code
        chroms.append(gray)  # append to chromosome list

    return chroms


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

    net = Net(n_feature=2, n_hidden=6, n_output=1)  # define the network

    # optional printouts about network
    print("printing net")
    print(net)  # net architecture

    weights = []
    weights = weightsOutofNetwork(net)
    # First layer of weights
    print(weights[0:18])
    # adjusting 3 weights in first layer and inserting them into the list
    weights = adjustweights(weights)
    net = weightsIntoNetwork(weights, net)

    # new weights after adjusting and adding back in
    weights = weightsOutofNetwork(net)

    print(weights[:18])

    print(real2Chrom(weights))



if __name__ == "__main__":
    main()
