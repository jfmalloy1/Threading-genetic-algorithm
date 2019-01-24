import numpy as np
import random
from deap import base, creator, tools

#Read in adj matrix and CPU cost
adj = []
cost = {}
#Read in adjanacy matrix as a 2D list of floats
with open("adj_matrix.txt") as a:
    for line in a:
        adj.append(line.split())
adj = np.array(adj)
adj = adj.astype(np.float)

#Read in CPU costs as a dictionary
with open("CPU_cost.txt") as c:
    next(c)
    for line in c:
        (key, val) = line.split()
        cost[int(key)] = float(val)

#Calculate the weight it takes to travel from n1 to n2
def weight(n1, n2):
    #use adj to calculate weight
    return adj[n1][n2]

#Calculate the distance between two nodes on a chip
def distance(n1, n2):
    row = abs(n1/8 - n2/8) #TODO: is this right?
    col = abs(n1 - n2%8)
    return row+col

#Calculate the fitness of an individual mapping
def fitness(individual):
    total = 0
    #go through each individual
    for i in individual:
        #find total energy consumption
        ec = 0
        #loop through individual twice
        for n in range(len(i)):
            for n2 in range(len(i)):
                #TODO is this right?
                #energy = cost to move * weight to move * distance for each
                ec += cost[i[n]] * weight(n, n2) * distance(n, n2)
        total += ec
    return total,

#Generate a dictionary of all neighbors using the adjacency matrix
def generate_neighbors():
    n = {}
    l = 0
    for line in adj:
        neighbors = []
        for i in range(len(line)):
            if line[i] != 0:
                neighbors.append(i)
        n[l] = neighbors
        l += 1
    return n

#Remove all occurances of x from neighbor list
def remove_neighbors(neighbors, x):
    if not(neighbors[x]):
        return []
    else:
        neighbor_list = neighbors[x]
        del neighbors[x]
        for item in neighbors:
            if x in neighbors[item]:
                neighbors[item].remove(x)
        return neighbor_list

#Find the node in n_list that has the least neighbors
def least_neighbors(neighbors, n_list):
    min = float("inf")
    n_smallest = 0
    for n in n_list:
        n_count = len(neighbors[n])
        if (n_count < min):
            min = n_count
            n_smallest = n
    return n_smallest

#Use edge recominbination to simulate crossing over
def cross(c1):
    #all neighbor dictionary (key - node, value - list of all neighbors)
    neighbors = generate_neighbors()
    #new chromosome
    c = []

    x = random.randint(0, len(c1[0]) - 1)
    while (len(c) != len(c1[0])):
        #TODO fill this in
        c.append(x)
        n_list = remove_neighbors(neighbors, x)
        #if x has no neighbors, pick another random node
        if (len(n_list) == 0):
            while (x not in c):
                x = random.randint(0, len(c1[0]) - 1)
        #otherwise, pick the neighbor with the fewest neighbors
        else:
            x = least_neighbors(neighbors, n_list)
    c1 = c

#Mutates an individual using the swap method
def mutate(i):
    #randomly select edges to swap
    r1 = random.randint(0, len(i[0]) - 1)
    r2 = random.randint(0, len(i[0]) - 1)
    #swap position r1 and r2
    temp = i[0][r1]
    i[0][r1] = i[0][r2]
    i[0][r2] = temp

#Sets up and runs the genetic algorithm
def run():
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    #Individual - randomly ordered array of 1-64
    toolbox.register("setup", random.sample, xrange(1,65), 64)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.setup, 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #Use the user-defined fitness function
    toolbox.register("evaluate", fitness)
    toolbox.register("select", tools.selTournament, tournsize = 2)

    #TODO: crossover and mutate
    toolbox.register("mutate", mutate)
    toolbox.register("crossover", cross)

    #population size
    pop = toolbox.population(n=50)
    fitnesses = list(map(toolbox.evaluate, pop))

    #Extract fitnesses
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    #crossing probability
    CXPB = 0.2
    #mutation probability
    MUTPB = 0.5

    #run for 10 generations initially
    g = 0
    while g < 100:
        g += 1
        print("-- Generation %i --" %g)

        #select the next gen individuals
        offspring = toolbox.select(pop, len(pop))
        #Clone selected offspring (IMPORTANT)
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.crossover(child1)
                del child1.fitness.values

        #Mutate selected individuals
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]

        #Calcualte summary statistics for each generation
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2/length - mean**2)**0.5

        print(" Min %s" % '{:0.3e}'.format(min(fits)))
        print(" Max %s" % '{:0.3e}'.format(max(fits)))
        print(" Avg %s" % '{:0.3e}'.format(mean))
        print(" Std %s" % '{:0.3e}'.format(std))


def main():
    run()

main()
