import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

# Code created using the following resources
#
#   https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35
#   https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)
#   https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm#:~:text=Fitness%20Proportionate%20Selection%20is%20one,features%20to%20the%20next%20generation.
#   https://mat.uab.cat/~alseda/MasterOpt/GeneticOperations.pdf
#

def create_graph(graph, n_nodes):
    ## O grafo sempre é o mesmo para alterar ele deve-se: 1) adicionar mais nodes 2) alterar a seed de aleatoriedade para os pesos do grafos
    rng = np.random.RandomState(123)
    random_ints = rng.randint(1, 30, size= n_nodes*n_nodes)
    nodes_idx = []

    for i in range(0, n_nodes):
        nodes_idx.append(i)

    graph.add_nodes_from(nodes_idx)

    weight_iter = 0

    for i in range(len(nodes_idx)):
        for j in range(len(nodes_idx)):
            graph.add_edges_from([(i, j)], weight=random_ints[weight_iter])
            weight_iter += 1


    return graph

def init_population(pop_number, gene_pool, state_length):
    population = []
    for i in range(pop_number):
        new_individual = np.random.choice(gene_pool, state_length, replace=False)
        new_individual = np.insert(new_individual, len(gene_pool), new_individual[0])
        population.append(new_individual)

    return population

def fitness_evaluation(G, state):
    total = 0
    for i in range(len(state) - 1):
        a = state[i]
        b = state[i+1]
        total += G[a][b]['weight']

    return total

def population_fitness(G, n_population, population):
    fitness_list = np.zeros(n_population)

    for i in range(n_population):
        fitness_list[i] =  fitness_evaluation(G, population[i])

    return fitness_list

def tournament_selection(G, population, tournament_size):
    tournament = random.sample(population, tournament_size)
    tournament_fitness = [fitness_evaluation(G, individual) for individual in tournament]

    return tournament[tournament_fitness.index(min(tournament_fitness))]


def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def mutate(individual, mutationRate):
    idx_1 = 0
    idx_2 = 0

    if(random.random() < mutationRate):
        
        while(idx_1 == idx_2):
            idx_1 = random.randint(1, len(individual)-2)
            idx_2 = random.randint(1, len(individual)-2)

        aux = individual[idx_1]
        individual[idx_1] = individual[idx_2]
        individual[idx_2] = aux

    return individual

def mutate_population(population, mutationRate):
    mutatedPop = []

    for idx in range(0, len(population)):
        mutatedIdx = mutate(population[idx], mutationRate)
        mutatedPop.append(mutatedIdx)

    return mutatedPop

def nextGeneration(G, population, mutationRate, breed_number,tournament_size):
    children = []
    for i in range(0, breed_number):
        selectionResults1 = tournament_selection(G, population, tournament_size)
        selectionResults2 = tournament_selection(G, population, tournament_size)
        children.append(breed(selectionResults1, selectionResults2))
    nextGeneration = mutate_population(children, mutationRate)
    return nextGeneration

def geneticAlgorithm(G, generations):
    
    n_population = 100
    mutation_rate = 0.10
    tounament_size = 25
    breed_number = n_population
    pop = init_population(n_population ,list(G.nodes), len(G.nodes))

    progress = []
    for i in tqdm(range(0, generations)):
        pop = nextGeneration(G, pop, mutation_rate, breed_number, tounament_size)
        progress.append(min(population_fitness(G, n_population, pop)))

    final_fitness = [fitness_evaluation(G, individual) for individual in pop]
    bestRoute = pop[final_fitness.index(min(final_fitness))]

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


    return bestRoute

def main():

    G = nx.Graph()
    G = create_graph(G, 250)  # NODES no grafo 
    print("graph created")
    #pos = nx.spring_layout(G, seed=20)
    #labels = {e: G.edges[e]['weight'] for e in G.edges}
    #nx.draw_networkx(G, pos=pos)
    #nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
    #plt.show()

    maxGen = 15

    bestRoute = geneticAlgorithm(G, maxGen)
    print('best route:', bestRoute)
    return 0

main()

