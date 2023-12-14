# General Libraries
import random as rand
import numpy as np

# Graph Representation & Embedding Library
import networkx as nx 

import differentGraphs as dg

from dataGenerationSAGE import KernighanLinIterationAndEmbedding
from writeToExcel import writeToExcel
from genEmbYesLabeledSAGE import yesLabeledEmbedding

### DATA GENERATION

TOTAL_NUMBER_OF_GRAPH_FOR_EACH = 10
NODES_LOW_LIMIT = 60
NODES_HIGH_LIMIT = 200

def dataGenerate(numberOfNodesLowest, numberOfNodesHighest):
    df=[]
    graphEmbedding = []
    seed = rand.randint(1,1000000)
    numberOfNodes = rand.randint(numberOfNodesLowest,numberOfNodesHighest)

    # Number 1 yesLabeledGraph
    # TODO Burada edges between partition'a da random'lık ekle
    i = TOTAL_NUMBER_OF_GRAPH_FOR_EACH*4
    while i>0:
        numberOfNodes = rand.randint(numberOfNodesLowest,numberOfNodesHighest)
        graphEmbedding = yesLabeledEmbedding(int(numberOfNodes/2), edgesBetweenPartitions=3)
        graphEmbedding = np.append(graphEmbedding, 1)
        if len(df) == 0:
            df = graphEmbedding
        else : 
            df = np.vstack((df, graphEmbedding))
        i = i-1

    i = TOTAL_NUMBER_OF_GRAPH_FOR_EACH    
    # Number 2 watts_strogatz
    while i>0:
        k_neighbors = rand.randint(2,3)
        probability = rand.random()
        numberOfNodes = rand.randint(numberOfNodesLowest,numberOfNodesHighest)
        G = dg.generate_watts_strogatz_graph(numberOfNodes,seed = seed, k_neighbors=k_neighbors, probability= probability)
        
        isValid = is_graph_appropriate(G)
        if isValid == False:
            continue

        G = make_graph_connected(G)
        
        # Iterate to find the best kernighan lin matching
        # TODO buraya matematiği getirilecek
        totalNumberOfIteration = 10
        didItBecomeConnected, graphEmbedding = KernighanLinIterationAndEmbedding(totalNumberOfIteration,G)

        if didItBecomeConnected and len(graphEmbedding)>0:
            i = i-1
            if len(df) == 0:
                df = graphEmbedding
            else : 
                df = np.vstack((df, graphEmbedding))


    i = TOTAL_NUMBER_OF_GRAPH_FOR_EACH
    # Number 3 Barabasi 
    while i>0:
        numberOfEdges = rand.randint(1,5)
        numberOfNodes = rand.randint(numberOfNodesLowest,numberOfNodesHighest)
        G = dg.generate_barabasi_albert_graph(numberOfNodes,seed = seed,edges=numberOfEdges)
        isValid = is_graph_appropriate(G)
        if isValid == False:
            continue

        G = make_graph_connected(G)
        totalNumberOfIteration = 10
        print(numberOfEdges)
        didItBecomeConnected, graphEmbedding = KernighanLinIterationAndEmbedding(totalNumberOfIteration,G)
        if didItBecomeConnected and len(graphEmbedding)>0:
            i = i-1
            if len(df) == 0:
                df = graphEmbedding
            else : 
                df = np.vstack((df, graphEmbedding))
    

    
    i = TOTAL_NUMBER_OF_GRAPH_FOR_EACH
    # Number 4 Erdos_renyi
    while i>0:
        probability = rand.random()%0.1
        numberOfNodes = rand.randint(numberOfNodesLowest,numberOfNodesHighest)
        G = dg.generate_erdos_renyi_graph(numberOfNodes,seed = seed, probability= probability)
        isValid = is_graph_appropriate(G)
        if isValid == False:
            continue

        G = make_graph_connected(G)
        totalNumberOfIteration = 10
        print(probability)
        didItBecomeConnected, graphEmbedding = KernighanLinIterationAndEmbedding(totalNumberOfIteration,G)
        if didItBecomeConnected and len(graphEmbedding)>0:
            i = i-1
            if len(df) == 0:
                df = graphEmbedding
            else : 
                df = np.vstack((df, graphEmbedding))
    
    

    i = TOTAL_NUMBER_OF_GRAPH_FOR_EACH
    # Number 5 Geometric_graph
    while i>0:
        radius = rand.random() % 0.05
        numberOfNodes = rand.randint(numberOfNodesLowest,numberOfNodesHighest)
        G = dg.generate_random_geometric_graph(numberOfNodes,seed = seed, radius=radius)
        isValid = is_graph_appropriate(G)
        if isValid == False:
            continue

        G = make_graph_connected(G)
        totalNumberOfIteration = 10
        print(radius)
        didItBecomeConnected, graphEmbedding = KernighanLinIterationAndEmbedding(totalNumberOfIteration,G)
        if didItBecomeConnected and len(graphEmbedding)>0:
            i = i-1
            if len(df) == 0:
                df = graphEmbedding
            else : 
                df = np.vstack((df, graphEmbedding))


    i = TOTAL_NUMBER_OF_GRAPH_FOR_EACH
    # Number 6 Planar_graph
    while i>0:
        numberOfNodes = rand.randint(numberOfNodesLowest,numberOfNodesHighest)
        numberOfEdges = rand.randint(int(numberOfNodes*1.15),int(numberOfNodes*1.3))
        G = dg.generate_planar_graph(nodes=numberOfNodes,edges=numberOfEdges)
        print(nx.number_of_edges(G))
        print(nx.number_of_nodes(G))
        while not nx.is_connected(G):
            numberOfNodes = rand.randint(numberOfNodesLowest,numberOfNodesHighest)
            numberOfEdges = rand.randint(int(numberOfNodes*1.15),int(numberOfNodes*1.3))
            G = dg.generate_planar_graph(nodes=numberOfNodes,edges=numberOfEdges)

        totalNumberOfIteration = 10
        didItBecomeConnected, graphEmbedding = KernighanLinIterationAndEmbedding(totalNumberOfIteration,G)
        if didItBecomeConnected and len(graphEmbedding)>0:
            i = i-1
            if len(df) == 0:
                df = graphEmbedding
            else : 
                df = np.vstack((df, graphEmbedding))

    i = TOTAL_NUMBER_OF_GRAPH_FOR_EACH
    # Number 7 Tree-like_graph
    while i>0:
        maxHeight = 5
        minHeight = 2
        maxBranch = 5
        minBranch = 2
        height = rand.randint(int(minHeight),int(maxHeight))
        numberOfBranches = rand.randint(int(minBranch),int(maxBranch))
        G = dg.generate_tree_graph(height=height,branches=numberOfBranches)
        print(nx.number_of_edges(G))
        print(nx.number_of_nodes(G))
        while not nx.is_connected(G):
            height = rand.randint(int(minHeight),int(maxHeight))
            numberOfBranches = rand.randint(int(minBranch),int(maxBranch))
            G = dg.generate_tree_graph(height=height,branches=numberOfBranches)

        totalNumberOfIteration = 10
        didItBecomeConnected, graphEmbedding = KernighanLinIterationAndEmbedding(totalNumberOfIteration,G)
        if didItBecomeConnected and len(graphEmbedding)>0:
            i = i-1
            if len(df) == 0:
                df = graphEmbedding
            else : 
                df = np.vstack((df, graphEmbedding))



    writeToExcel(df)
    
    
    
def make_graph_connected(G):
    
    while not nx.is_connected(G):
        # Randomly select two nodes from the graph
        node1 = rand.choice(list(G.nodes()))
        node2 = rand.choice(list(G.nodes()))

        # Add an edge between the selected nodes
        G.add_edge(node1, node2)
        
    return G

# Gets a graph and makes is connected
def is_graph_appropriate(graph):
    # Check if the graph is None
    if graph is None:
        return False

    # Check if the graph is empty
    if len(graph.nodes()) == 0:
        return False
    
    return True