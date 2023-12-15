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

def KernighanLinIterationAndEmbedding(totalNumberOfIteration, G):
    didItBecomeConnected = False
    graphEmbedding = []
    for j in range(totalNumberOfIteration):
        partition = kernighan_lin_bisection(G,max_iter = 1000)
        G_partition1 = G.subgraph(partition[0])
        G_partition2 = G.subgraph(partition[1])
        if nx.is_connected(G_partition1) and nx.is_connected(G_partition2):
            didItBecomeConnected = True
            total_edges = G.number_of_edges()
            partition_1_edges = G_partition1.number_of_edges()
            partition_2_edges = G_partition2.number_of_edges()
            edgeBetweenSubGraphs = total_edges-partition_1_edges-partition_2_edges
            # Buradaki maksimum edge belirlenecek
            # TODO matemetigi getirilecek
            print(str(edgeBetweenSubGraphs) + "        " + str(j))
            if edgeBetweenSubGraphs < 7: 
                print('girdi1')   
                nodeEmbeddings = getEmbedding(G) 
                nodeEmbeddingsArray = dictionaryToNpArray(nodeEmbeddings)
                graphEmbedding = np.mean(nodeEmbeddingsArray, axis=0)
                graphEmbedding = np.append(graphEmbedding, 1)
                
                #np.append(graphEmbedding, findArticulationPoints(G))
                #np.append(graphEmbedding, calculate_density(G))
                
                break
            
        # Sona geldiysek ve hala Yes label alamadıysa No label ver
        # Eger hicbir zaman connected bir sekilde bolunemediyse hicbir sey yapma
        if j == totalNumberOfIteration-1 and didItBecomeConnected:
            print('girdi2')
            nodeEmbeddings = getEmbedding(G) 
            nodeEmbeddingsArray = dictionaryToNpArray(nodeEmbeddings)
            graphEmbedding = np.mean(nodeEmbeddingsArray, axis=0)
            graphEmbedding = np.append(graphEmbedding, 0)
            
            #np.append(graphEmbedding, findArticulationPoints(G))
            #np.append(graphEmbedding, calculate_density(G))
            
            break
    return didItBecomeConnected,graphEmbedding    