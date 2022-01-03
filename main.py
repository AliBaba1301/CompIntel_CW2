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
generations = 10000
cxPB = 0.7
loss = nn.MSELoss()
flipPB = 1 / (dimensions * numOfBits)
mutatePB = 0.2
nElitists = 1
dspInterval = 1


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


# creating dataset in order to train and test the model
# creating 1100 random values for x1 and x2 between -1 and 1
x1arr = np.random.uniform(-1, 1, 1100)
x2arr = np.random.uniform(-1, 1, 1100)

# creating a list of the function values
yarr = function(x1arr, x2arr)

# create a 3d plot of the function
visualfunction(x1arr, x2arr, 'All Data')

# creating a list of the training data
input = np.array([x1arr, x2arr])
inputTensor = torch.from_numpy(input)
inputTensor = torch.transpose(inputTensor, 1, 0)

target = np.array(yarr)
targetTensor = torch.from_numpy(target)
main_net = Net(n_feature=2, n_hidden=6, n_output=1)


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
    numinrange = (minRange + (maxRange - minRange) * numasint / maxnum)
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
        if weights[i] >= maxRange:
            weights[i] = int(maxRange)
        elif weights[i] <= minRange:
            weights[i] = int(minRange)

        # rounding value to prevent overflow and value between 0 and 1
        numPrepped = ((weights[i] + maxRange) / (maxRange - minRange) * maxnum)
        # convert the integer part to binary
        integer = bin(int(numPrepped))[2:]

        # pad the binary with 0's to make it the correct length
        integer = integer.zfill(numOfBits)
        integer = integer
        # combine the two parts into one string
        numInBits = integer  # convert value to base 2
        gray = bin_to_gray(numInBits)  # convert to gray code
        if (len(gray) > numOfBits):
            gray = gray[1:]
        chroms.append(gray)  # append to chromosome list
        chroms = list(''.join(chroms))

    return chroms


# an evaluation function to return the mean squared error of the network on the data provided for a given generation
def evaluate(ind):
    # convert chromosomes to real numbers
    chroms = reformList(ind, 67, 30)
    weights = []
    for i in chroms:
        weights.append(chrom2real(i))

    weights = np.asarray(weights)
    # set the weights of the network to the weights in the chromosome
    weightsIntoNetwork(weights, main_net)

    # get the output of the network
    output = main_net(inputTensor[:1000])
    test_output = main_net(inputTensor[1000:1100])

    # calculate the error
    error = loss(output.reshape(-1), targetTensor[:1000])
    test_error = loss(test_output.reshape(-1), targetTensor[1000:1100])

    return error.item(),


def evaluateTest(ind):
    # convert chromosomes to real numbers
    chroms = reformList(ind, 67, 30)
    weights = []
    for i in chroms:
        weights.append(chrom2real(i))

    weights = np.asarray(weights)
    # set the weights of the network to the weights in the chromosome
    weightsIntoNetwork(weights, main_net)

    # get the output of the network
    test_output = main_net(inputTensor[1000:1100])

    # calculate the error
    error = loss(test_output.reshape(-1), targetTensor[1000:1100])

    return error.item(),


def plotFitness(loss_list, generation):
    plt.plot(generation, loss_list)
    plt.show()


def nn3dSurface(chrom):
    chrom = reformList(chrom, 67, 30)
    weights = []
    for i in chrom:
        weights.append(chrom2real(i))

    weights = np.asarray(weights)
    # set the weights of the network to the weights in the chromosome
    weightsIntoNetwork(weights, main_net)

    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    x1, x2 = np.meshgrid(x, y)

    xy = np.array([x, y])
    xyTensor = torch.from_numpy(xy)
    xyTensor = torch.transpose(xyTensor, 1, 0)

    nnYArr = main_net(xyTensor)
    nnYArr = nnYArr.detach().numpy()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # create the surface
    surfaceplot = ax.plot_surface(x1, x2, nnYArr, rstride=1, cstride=1, cmap=cm.viridis, alpha=0.8, linewidth=0,
                                  antialiased=False)
    ax.view_init(45, 45)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title('3D Neural Network Surface Plot')
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surfaceplot, shrink=0.5)
    plt.show()

# method for implementing Lamarckian learning
def lla(ind):
    # convert chromosomes to real numbers
    chroms = reformList(ind, 67, 30)
    weights = []
    for i in chroms:
        weights.append(chrom2real(i))

    weights = np.asarray(weights)
    # set the weights of the network to the weights in the chromosome
    weightsIntoNetwork(weights, main_net)

    # get the output of the network
    original_output = main_net(inputTensor[:1000])
    # get error for the original weights
    current_error = loss(original_output.reshape(-1), targetTensor[:1000])

    # grab the weights from the network
    new_weights = weightsOutofNetwork(main_net)

    optimizer = torch.optim.Rprop(main_net.parameters(), lr=0.003)
    j = 0
    while j < 30:  # run 30 iterations of optimization to find better weights
        new_out = main_net(inputTensor[:1000])  # input training data and predict network output based on data
        new_error = loss(new_out, targetTensor[:1000])  # compare output with labeled data
        optimizer.zero_grad()  # clear gradients for next train
        new_error.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        if current_error > new_error:  # update weights if loss result is better
            new_weights = weightsOutofNetwork(main_net)
            current_error = new_error
        j += 1
    # convert weights to a new gray coded individual
    newInd = real2Chrom(new_weights)
    return newInd

# method for implementing Baldwinian learning
def bla(ind):
    # convert chromosomes to real numbers
    chroms = reformList(ind, 67, 30)
    weights = []
    for i in chroms:
        weights.append(chrom2real(i))

    weights = np.asarray(weights)
    # set the weights of the network to the weights in the chromosome
    weightsIntoNetwork(weights, main_net)

    # get the output of the network
    original_output = main_net(inputTensor[:1000])
    # get error for the original weights
    current_error = loss(original_output.reshape(-1), targetTensor[:1000])

    # grab the weights from the network
    new_weights = weightsOutofNetwork(main_net)

    optimizer = torch.optim.Rprop(main_net.parameters(), lr=0.003)
    j = 0
    while j < 30:  # run 30 iterations of optimization to find better weights
        new_out = main_net(inputTensor[:1000])  # input training data and predict network output based on data
        new_error = loss(new_out, targetTensor[:1000])  # compare output with labeled data
        optimizer.zero_grad()  # clear gradients for next train
        new_error.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        if current_error > new_error:  # update weights if loss result is better
            new_weights = weightsOutofNetwork(main_net)
            current_error = new_error
        j += 1
    # convert weights to a new gray coded individual

    return current_error.item(),

def plotBlavsLla(blaList, llaList,title):
    plt.plot(blaList, 'r', label='Baldwinian')
    plt.plot(llaList, 'b', label='Lamarckian')
    title = 'Baldwinian vs Lamarckian Learning ' + title
    plt.title(title)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
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
                 toolbox.attr_bool, dimensions * numOfBits)

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
    popbla = pop
    gen = []
    best_fitness = []
    best_fitness_bla = []
    test_score = []
    test_score_bla = []

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    fitnesses_bla = list(map(toolbox.evaluate, popbla))
    for ind, fit in zip(popbla, fitnesses_bla):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]
    bestInitial = tools.selBest(pop, 1)[0].fitness.values[0]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while g < generations:
        # A new generation
        gen.append(g)
        g = g + 1
        print("-- Generation %i --" % g)

        # apply lamarckian evolution to the population
        for ind in pop:
            ind = lla(ind)
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # apply baldwinian evolution to the population
        fitnesses_bla = list(map(bla, popbla))
        for ind, fit_bla in zip(pop, fitnesses_bla):
            ind.fitness.values = fit_bla

        # Select the next generation individuals
        offspring = tools.selBest(pop, nElitists) + toolbox.select(pop, len(pop) - nElitists)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Selecting the best individual from the population
        best_ind = tools.selBest(pop, 1)[0]
        fitnessForBestIndividual = best_ind.fitness.values
        # adding fitness of best individual to list
        best_fitness.append(fitnessForBestIndividual)
        # adding test score of best individual to list
        test_score.append(evaluateTest(best_ind))

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
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Select the next generation individuals
        offspring_bla = tools.selBest(popbla, nElitists) + toolbox.select(popbla, len(popbla) - nElitists)
        # Clone the selected individuals
        offspring_bla = list(map(toolbox.clone, offspring_bla))

        # Selecting the best individual from the population
        best_ind_bla = tools.selBest(popbla, 1)[0]
        fitnessForBestBlaIndividual = best_ind_bla.fitness.values
        # adding fitness of best individual to list
        best_fitness_bla.append(fitnessForBestBlaIndividual)
        # adding test score of best individual to list
        test_score_bla.append(evaluateTest(best_ind_bla))

        # Apply crossover and mutation on the offspring
        # make pairs of offspring for crossing over
        for child1, child2 in zip(offspring_bla[::2], offspring_bla[1::2]):

            # cross two individuals with probability CXPB
            if r.random() < cxPB:
                # print('before crossover ',child1, child2)
                toolbox.mate(child1, child2)
                # print('after crossover ',child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant_bla in offspring_bla:
            # mutate an individual with probability mutateprob
            if r.random() < mutatePB:
                toolbox.mutate(mutant_bla)
                del mutant_bla.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring_bla if not ind.fitness.valid]
        fitnesses_bla = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses_bla):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        popbla[:] = offspring_bla

        if g % dspInterval == 0:
            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)

    # plot the fittest chromosome of each generation
    plotFitness(best_fitness, gen)
    print(min(best_fitness))

    # plot the test score of each generation
    plotFitness(test_score, gen)
    print(min(test_score))

    nn3dSurface(best_ind)

    plotFitness(best_fitness_bla, gen)
    print(min(best_fitness_bla))

    # plot the test score of each generation
    plotFitness(test_score_bla, gen)
    print(min(test_score_bla))

    nn3dSurface(best_ind_bla)

    plotBlavsLla(best_fitness_bla, best_fitness, 'Training')
    plotBlavsLla(test_score_bla, test_score, 'Test')


    # test the real2chrom method
    # r2cInput = weightsOutofNetwork(main_net)
    r2cInput = [6, 18, -19.2323, -20.233, -24, 19.9999999, 23, 20.00001]
    r2cInput = np.asarray(r2cInput)
    output_chrom = real2Chrom(r2cInput)
    test_chrom = reformList(output_chrom, len(r2cInput), 30)
    output_weights = []

    for i in test_chrom:
        output = chrom2real(i)
        output_weights.append(output)

    print("chromosome: ", output_chrom)
    print("test input: ", [6, 18, -19.2323, -20.233, -24, 19.9999999, 23, 20.00001])
    print("test output: ", output_weights)


if __name__ == "__main__":
    main()
