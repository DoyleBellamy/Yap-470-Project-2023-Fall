import networkx as nx
import random as rand
import numpy as np

from dataEmbeddingSAGE import getSAGEembedding

SEED_OF_RANDOM = 2524
rand.seed(SEED_OF_RANDOM)

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

    graphEmbedding = getSAGEembedding(G_union)
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

