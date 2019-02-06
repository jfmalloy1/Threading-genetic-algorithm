from __future__ import print_function
import numpy as np
import random
import csv
import sys
import collections
from deap import base, creator, tools
#from termcolor import colored

#Global population and generation variables
GEN = int(sys.argv[1])
POPULATION = int(sys.argv[2])
#chip size (one dimension)
CHIP = int(sys.argv[3])
#crossing probability
CXPB = float(sys.argv[4])
#mutation probability
MUTPB = float(sys.argv[5])
#Run number
#RUN = int(sys.argv[6])

#Read in adj matrix and CPU cost
adj = []
cost = {}
#Read in adjanacy matrix as a 2D list of floats
with open("adj_matrix.txt") as a:
    for line in a:
        adj.append(line.split())
adj = np.array(adj)
adj = adj.astype(np.float)

#open a csv output file
c = open("output" + str(GEN) + "_" + str(POPULATION) + "_" + str(CHIP) + "_" + str(CXPB) + "_" + str(MUTPB) + ".csv", "w")
fieldnames = ["Gen", "Min", "Max", "Mean"]
writer = csv.DictWriter(c, fieldnames = fieldnames)
writer.writeheader()

#open min values output file
c2 = open("minvalues" + str(GEN) + "_" + str(POPULATION) + "_" + str(CHIP) + "_" + str(CXPB) + "_" + str(MUTPB) + ".csv", "a")

#Calculate the weight it takes to travel from n1 to n2
def weight(n1, n2):
    #use adj to calculate weight
    return adj[n1][n2]

#Calculate the distance between two nodes on a chip
def distance(n1, n2):
    row = abs(n1/CHIP - n2/CHIP)
    col = abs(n1 - n2%CHIP)
    return row+col

#Calculate the fitness of an individual mapping
def fitness(individual):
    total = 0
    #print individual
    #go through each individual
    for i in individual:
        #find total energy consumption
        ec = 0
        #loop through individual twice
        for n in range(len(i)):
            for n2 in range(len(i)):
                #energy = weight to communicate with entire chip * distance for each
                ec += weight(i[n]-1, i[n2]-1) * distance(n, n2)
        total += ec
    return total,

#Make a dictionary of all adjacent edges
def make_adj_dict(c1, c2):
    a = {}
    for i in range(1, len(c1) + 1):
        a[i] = []

    for i in range(0, len(c1)):
        l =  a[c1[i]]
        if (i - 1 >= 0):
            l.append(c1[i-1])
        if (i + 1 <= (CHIP**2) - 1):
            l.append(c1[i+1])
        a[c1[i]] = l

    for i in range(0, len(c2)):
        l =  a[c2[i]]
        if (i - 1 >= 0):
            l.append(c2[i-1])
        if (i + 1 <= (CHIP**2) - 1):
            l.append(c2[i+1])
        a[c2[i]] = l

    return a

#Pick the next edge to append (for edge recombination)
def pick_next(offspring, a):
    #find repeating edges
    repeat = [item for item, count in collections.Counter(a[offspring[-1]]).items() if count > 1]
    #if there are repeating edges, pick one and return it
    if (len(repeat) > 0):
        for n in repeat:
            if (n not in offspring):
                return n
    #if no repeating edges, but edges exist that are not already added, add them
    if (a[offspring[-1]] != None):
        for n in a[offspring[-1]]:
            if n not in offspring:
                return n
    #if no edges, pick a random number from those left over
    all_nums = range(1, CHIP**2+1)
    not_shown = list(set(all_nums) - set(offspring))
    return random.choice(not_shown)

#Use edge recominbination to simulate crossing over
def cross(c1, c2):
    #generate adjanacy list
    adj_dict = make_adj_dict(c1[0], c2[0])

    #populate offspring
    offspring = []
    while (len(offspring) != len(c1[0])):
        #randomly begin
        if (offspring == []):
            offspring.append(random.randint(1, CHIP**2))
        else:
            #keep edge recominbination
            offspring.append(pick_next(offspring, adj_dict))

    c1[0] = offspring
    c2[0] = offspring

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

    #Individual - randomly ordered array of 1-chip size (determined by sys args)
    toolbox.register("setup", random.sample, xrange(1,(CHIP**2 + 1)), CHIP**2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.setup, 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #Use the user-defined fitness function
    toolbox.register("evaluate", fitness)
    toolbox.register("select", tools.selTournament, tournsize = 2)

    #crossover and mutate
    toolbox.register("mutate", mutate)
    toolbox.register("crossover", cross)

    #population size
    pop = toolbox.population(n=POPULATION)
    fitnesses = list(map(toolbox.evaluate, pop))

    #Extract fitnesses
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    #run for 10 generations initially
    g = 0
    while g < GEN:
        g += 1
        print("-- Generation %i --" %g)

        #select the next gen individuals
        offspring = toolbox.select(pop, len(pop))
        #Clone selected offspring (IMPORTANT)
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.crossover(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

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

        writer.writerow({"Gen": g, "Min": min(fits), "Max": max(fits), "Mean": mean})

        #find the min value, print it to an output file
        fmin = min(fits)
        min_pos = [i for i,x in enumerate(fits) if x == fmin]
        #choose one of the minimum fitness values (seems to be the same mapping)
        c = random.choice(min_pos)

        #output each min mapping along with fitness
        for n in pop[c][0]:
            print(str(n), end=' ', file=c2)
        print('',file=c2)
        print(str(min(fits)), file=c2)

def main():
    run()

main()
