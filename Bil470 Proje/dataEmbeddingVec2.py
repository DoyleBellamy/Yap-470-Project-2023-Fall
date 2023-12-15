# General Libraries
import numpy as np
import pandas as pd

# Graph Representation & Embedding Library
import networkx as nx 

# Kernigan Algorithm Library
from kernighan_lin import kernighan_lin_bisection

from node2vec import Node2Vec

from getExtraGraphFeatures import findArticulationPoints, calculate_density


def getEmbedding(G):
    # Precompute probabilities and generate walks
    node2vec = Node2Vec(G, dimensions=30, walk_length=10, num_walks=40, workers=4)

    # Embed nodes
    model = node2vec.fit(window=5, min_count=1, batch_words=4)

    # Get embeddings for all nodes in the graph
    all_node_embeddings = {node: model.wv[str(node)] for node in G.nodes()}

    return all_node_embeddings
    # loaded_model = Node2Vec.load("node2vec_model.bin")
    # loaded_embedding = loaded_model.wv['0']
    # print(f"Loaded Embedding for Node 0: {loaded_embedding}")


# TODO
def writeEmbeddingToFile(embeddingList):
    return 0

def dictionaryToNpArray(embedding):
    array_from_dict = np.array(list(embedding.values()))
    return array_from_dict

def KernighanLinIterationAndEmbedding(G):

    graphEmbedding = []
    
    total_vertices = G.number_of_nodes()
    
    # determine total number of iterations
    if total_vertices < 360:
        totalNumberOfIteration = 10
    elif total_vertices > 500:
        totalNumberOfIteration = 0.4 * total_vertices * np.log10(total_vertices)
    else:
        totalNumberOfIteration = total_vertices * np.log10(total_vertices)
    
    totalNumberOfIteration = int(totalNumberOfIteration)
    
    #for j in range(totalNumberOfIteration):
    partition = kernighan_lin_bisection(G, max_iter=totalNumberOfIteration)
    
    G_partition1 = G.subgraph(partition[0])
    G_partition2 = G.subgraph(partition[1])
    
    if nx.is_connected(G_partition1) and nx.is_connected(G_partition2):
            
        # check vertex constraint
        partition_1_vertices = G_partition1.number_of_nodes()
        partition_2_vertices = G_partition2.number_of_nodes()
        
        min_vertex_bound = total_vertices/2 - total_vertices*0.01
        max_vertex_bound = total_vertices/2 + total_vertices*0.01
        
        if ((min_vertex_bound <= partition_1_vertices <= max_vertex_bound)
                and (min_vertex_bound <= partition_2_vertices <= max_vertex_bound)):
            
            print('girdi1')   
            nodeEmbeddings = getEmbedding(G) 
            nodeEmbeddingsArray = dictionaryToNpArray(nodeEmbeddings)
            graphEmbedding = np.mean(nodeEmbeddingsArray, axis=0)
            graphEmbedding = np.append(graphEmbedding, 1)
        
        else:
            print('girdi2')   
            nodeEmbeddings = getEmbedding(G) 
            nodeEmbeddingsArray = dictionaryToNpArray(nodeEmbeddings)
            graphEmbedding = np.mean(nodeEmbeddingsArray, axis=0)
            graphEmbedding = np.append(graphEmbedding, 0)
    else:
        print('girdi3')   
        nodeEmbeddings = getEmbedding(G) 
        nodeEmbeddingsArray = dictionaryToNpArray(nodeEmbeddings)
        graphEmbedding = np.mean(nodeEmbeddingsArray, axis=0)
        graphEmbedding = np.append(graphEmbedding, 0)

    return graphEmbedding    