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
from deap import base, creator, tools, algorithms

torch.manual_seed(1)  # reproducible experiments

numOfBits = 30
dimensions = 67
maxnum = (2 ** numOfBits)
minRange = -20
maxRange = 20
generations = 1000
cxPB = 0.4
loss = nn.MSELoss()
flipPB = 1/(dimensions*numOfBits)
mutatePB = 0.1
nElitists = 1


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
        x = nn.functional.sigmoid(self.hidden2(x))
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
    ax.set_zlabel('y')
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

    for i in range(len(weights)):
        for j in range(len(weights[i])):
            weightlist.append(weights[i][j])

    return weightlist


# takes as in a list of all weights of the network and uses them to set the weights of the network
def weightsIntoNetwork(weights, net):
    net.hidden.weight = torch.nn.Parameter(torch.from_numpy(reformList(weights[:12], 6, 2)))
    net.hidden.bias = torch.nn.Parameter(torch.from_numpy(reformList(weights[12:18], 1, 6)))
    net.hidden2.weight = torch.nn.Parameter(torch.from_numpy(reformList(weights[18:54], 6, 6)))
    net.hidden2.bias = torch.nn.Parameter(torch.from_numpy(reformList(weights[54:60], 1, 6)))
    net.out.weight = torch.nn.Parameter(torch.from_numpy(reformList(weights[60:66], 1, 6)))
    net.out.bias = torch.nn.Parameter(torch.from_numpy(reformList(weights[66:67], 1, 1)))


# Convert chromosome to real number
# input: list binary 1,0 of length numOfBits representing number using gray  coding
# output: real value of chromosome
def chrom2real(c):
    indasstring = ''.join(map(str, c))
    degray = gray_to_bin(indasstring)
    numasint = int(degray, 2)  # convert to int from base 2 list
    numinrange = round(minRange + (maxRange - minRange) * numasint / maxnum, 7)
    # ensures the weights are in the range of -20 and 20
    if numinrange > maxRange:
        numinrange = maxRange
    elif numinrange < minRange:
        numinrange = minRange

    return numinrange


# converts a list of weights into a list with the weights as chromosomes
# range of weights is between -20 and 20
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

        # rounding value to prevent overflow and value between 0 and 1
        numPrepped = round(((weights[i] + maxRange) / (maxRange - minRange) * maxnum))

        # convert the integer part to binary
        integer = bin(int(numPrepped))[2:]

        # pad the binary with 0's to make it the correct length
        integer = integer.zfill(numOfBits)

        # combine the two parts into one string
        numInBits = integer  # convert value to base 2
        gray = bin_to_gray(numInBits)  # convert to gray code
        chroms.append(gray)  # append to chromosome list

    return chroms


# an evaluation function to return the mean squared error of the network on the data provided for a given generation
def evaluate(net, input, target,ind):
    # convert chromosomes to real numbers
    chroms = reformList(ind, 67, 30)
    weights = []

    for i in range(len(chroms)):
        weights.append(chrom2real(chroms[i]))

    # set the weights of the network to the weights in the chromosome
    weightsIntoNetwork(weights, net)

    # get the output of the network
    output = net(input)

    # calculate the error
    error = loss(output, target)

    return error.item(),


def plotFitness(loss_list, generation):
    plt.plot(generation, loss_list)
    plt.show()


toolbox = base.Toolbox()
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Attribute generator
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
toolbox.register("attr_bool", r.randint, 0, 1)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of numOfBitsdimension 'attr_bool' elements ('genes')
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, 2010)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ----------
# Operator registration
# ----------
# register the goal / fitness function
toolbox.register("evaluate", evaluate)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=flipPB)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selBest, fit_attr='fitness')


# ----------


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
    inputTraining = np.array([x1arr[:1000], x2arr[:1000]])
    inputTrainingTensor = torch.from_numpy(inputTraining)
    inputTrainingTensor = torch.transpose(inputTrainingTensor, 1, 0)

    targetTraining = np.array(yarr[:1000])
    targetTrainingTensor = torch.from_numpy(targetTraining)

    # Not needed anymore # plot the training data as a 3D scatter plot
    # scatter3D(x1arrtrain, x2arrtrain, yarrtrain, title='Training Data')

    # creating a list of the testing data
    inputTesting = np.array([x1arr[1000:1100], x2arr[1000:1100]])
    inputTestingTensor = torch.from_numpy(inputTesting)
    inputTestingTensor = torch.transpose(inputTestingTensor, 1, 0)

    targetTesting = np.array(yarr[1000:1100])
    targetTestingTensor = torch.from_numpy(targetTesting)

    # Not needed anymore # plot testing data as a 3D scatter plot
    # scatter3D(x1arrtest, x2arrtest, yarrtest, title='Testing Data')

    net = Net(n_feature=2, n_hidden=6, n_output=1)  # define the network

    # optional printouts about network
    print("printing net")
    print(net)  # net architecture

    weights = []
    weights = weightsOutofNetwork(net)

    # First layer of weights
    # adjusting 3 weights in first layer and inserting them into the list
    weights = adjustweights(weights)

    weightsIntoNetwork(weights, net)

    # new weights after adjusting and adding back in
    weights = weightsOutofNetwork(net)

    # initialize the population
    initial_weights = weightsOutofNetwork(net)

    # genetic algorithm for training the network
    # creating a population of chromosomes

    pop = toolbox.population(n=100)  # Population Size
    new_net = Net(n_feature=2, n_hidden=6, n_output=1)
    fitness = []
    gen = []
    best_fitness = []
    test_score = []

    for g in range(generations):

        print("Generation: %i" %g)
        gen.append(g)
        for i in range(len(pop)):
            # train the network on the training data
            fitness.append(toolbox.evaluate(new_net, inputTrainingTensor, targetTrainingTensor,pop[i])[0])

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop) - nElitists) + tools.selBest(pop, nElitists)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        #         for individ in offspring:
        #             print(individ)

        # Apply crossover and mutation on the offspring
        # make pairs of offspring for crossing over
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if r.random() < cxPB:
                # print('before crossover ',child1, child2)
                toolbox.mate(child1, child2)
                # print('after crossover ',child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability mutateprob
            if r.random() < mutatePB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.value = toolbox.evaluate(new_net, inputTrainingTensor, targetTrainingTensor,ind)[0]

        # best unique individual
        best_ind = tools.selBest(pop, 1)[0]
        # The population is entirely replaced by the offspring
        pop[:] = offspring

        best_fitness.append(min(fitness))

        # adding in test data
        score = evaluate(new_net, inputTestingTensor, targetTestingTensor, best_ind)[0]
        test_score.append(score)





    # plot the fittest chromosome of each generation
    plotFitness(best_fitness, gen)

    # plot the test score of each generation
    plotFitness(test_score, gen)




if __name__ == "__main__":
    main()
