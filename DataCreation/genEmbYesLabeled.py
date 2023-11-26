import networkx as nx
import random as rand
import numpy as np
import matplotlib.pyplot as plt
from kernighan_lin import kernighan_lin_bisection
from Node2Vec_Part.node2vec_self_imp  import getEmbedding

SEED_OF_RANDOM = 2524
rand.seed(SEED_OF_RANDOM)

# Burada aslinda herhangi bir kontrol yapmaya gerek yok
# Kernighan-Lin falan kaldirilacak

def yesLabeledEmbedding(totalNumberOfNodes = 100, edgesBetweenPartitions = 3):
    NUMBER_OF_NODES = totalNumberOfNodes
    nodeList_1 = np.arange(1,NUMBER_OF_NODES+1)
    nodeList_2 = np.arange(NUMBER_OF_NODES+1,2*NUMBER_OF_NODES+1)
    connected_graph_1 = create_connected_graph(nodeList_1)
    connected_graph_2 = create_connected_graph(nodeList_2)
    RENAME_1 = 'G_1_' 
    RENAME_2 = 'G_2_'
    G_union = nx.union(connected_graph_1,connected_graph_2,rename=(RENAME_1,RENAME_2))
    EDGES_BETWEEN_PARTITIONS = edgesBetweenPartitions
    for i in range (EDGES_BETWEEN_PARTITIONS):
        node1 = RENAME_1 + str(rand.randint(1,NUMBER_OF_NODES))
        node2 = RENAME_2 + str(rand.randint(NUMBER_OF_NODES+1,2*NUMBER_OF_NODES+1))
        G_union.add_edge(node1,node2)
    # partition = kernighan_lin_bisection(G_union,max_iter = 200)
    # G_partition1 = G_union.subgraph(partition[0])
    # G_partition2 = G_union.subgraph(partition[1])
    # # total_edges = G_union.number_of_edges()
    # partition_1_edges = G_partition1.number_of_edges()
    # partition_2_edges = G_partition2.number_of_edges()
    nodeEmbeddings = getEmbedding(G_union) 
    nodeEmbeddingsArray = dictionaryToNpArray(nodeEmbeddings)
    graphEmbedding = np.mean(nodeEmbeddingsArray, axis=0)
    return graphEmbedding

def create_connected_graph(node_list):
    # Create an empty graph
    G = nx.Graph()

    # Add nodes from the specific node list
    G.add_nodes_from(node_list)

    # Add edges to connect the nodes and ensure connectivity
    while not nx.is_connected(G):
        # Randomly select two nodes from the graph
        node1 = rand.choice(list(G.nodes()))
        node2 = rand.choice(list(G.nodes()))

        # Add an edge between the selected nodes
        G.add_edge(node1, node2)

    return G

def dictionaryToNpArray(embedding):
    array_from_dict = np.array(list(embedding.values()))
    return array_from_dict
