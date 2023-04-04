import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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

def generate_successors(G, state):
    successor_states = []
    for i in range(1, len(state) - 2):
        successor = state.copy()

        # swap adjacent
        aux = successor[i]
        successor[i] = successor[i+1]
        successor[i+1] = aux
        successor_states.append(successor)

    return successor_states

def beam_search(G, k, n_generations):
    states = init_population(k, list(G.nodes), len(G.nodes))
    states = sorted(states, key=lambda x: fitness_evaluation(G, x))

    progress = []
    for i in tqdm(range(0, n_generations)):
        all_successors = []
        for state in states:
            sucessors_states = generate_successors(G, state)
            all_successors.extend(sucessors_states)

        all_successors = sorted(all_successors, key=lambda x: fitness_evaluation(G, x))

        states = all_successors[:k]

        best_fitness = fitness_evaluation(G, states[0])

        progress.append(best_fitness)


    #plt.plot(progress)
    #plt.ylabel('Distance')
    #plt.xlabel('Generation')
    #plt.show()

    return states[0]



def main():

    G = nx.Graph()
    G = create_graph(G, 500)  # NODES no grafo 
    
    print("graph created")

    #pos = nx.spring_layout(G, seed=20)
    #labels = {e: G.edges[e]['weight'] for e in G.edges}
    #nx.draw_networkx(G, pos=pos)
    #nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
    #plt.show()

    k = 10
    n_generations = 100
    best_route = beam_search(G, k, n_generations)
    print(best_route)
    # plt.show()

    return 0

main()