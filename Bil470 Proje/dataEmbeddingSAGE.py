# General Libraries
import numpy as np
import pandas as pd

# Graph Representation & Embedding Library
import networkx as nx 

# Kernigan Algorithm Library
from kernighan_lin import kernighan_lin_bisection

# GrapSAGE Libraries
from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from stellargraph.layer import MeanPoolingAggregator 
from tensorflow.keras import layers, optimizers, losses, Model

from getExtraGraphFeatures import findArticulationPoints, calculate_density

# Get degree feature
def get_degree_feature(graph):
    return dict(graph.degree())

# Get betweenness centrality feature
def get_betweenness_centrality_feature(graph):
    return nx.betweenness_centrality(graph)

# Get Average Clustering
def get_average_clustring(G):
    return nx.average_clustering(G)


# Generate node features
def generate_node_features(graph, feature_functions):
    feature_dict = {}
    
    # Calculate each feature using the provided functions
    for feature_name, feature_function in feature_functions.items():
        feature_dict[feature_name] = feature_function(graph)
    
    # Create a DataFrame with node features
    node_features = pd.DataFrame(feature_dict, index=graph.nodes())
    
    return node_features


def getSAGEembedding(G):
    
    feature_functions = {
        'degree': get_degree_feature,
        'betweenness_centrality': get_betweenness_centrality_feature,
        'get_average_clustring': get_average_clustring,
        # Add more features as needed
    }
    
    # Assuming you have a function to generate node features, modify as needed
    node_features = generate_node_features(G, feature_functions)
    
    # Convert NetworkX graph to StellarGraph with node features
    G_stellar = StellarGraph.from_networkx(G, node_features=node_features)

    # generator
    # batch_size -> number of nodes per batch
    # num_samples -> number of neighbours per layer
    generator = GraphSAGENodeGenerator(G_stellar, batch_size=50, num_samples=[10, 10])
    
    model = GraphSAGE(layer_sizes=[50, 50], generator=generator, aggregator=MeanPoolingAggregator, bias=True, dropout=0.5)
    
    # get input and output tensors
    x_inp, x_out = model.in_out_tensors()

    output_size = 30
    
    # pass the output tensor through the classification layer
    prediction = layers.Dense(units=output_size, activation="linear")(x_out)

    # Combine the GraphSAGE model with the prediction layer
    model = Model(inputs=x_inp, outputs=prediction)

    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=1e-3), loss=losses.binary_crossentropy, metrics=["acc"])

    #model.summary()
    
    # Obtain the graph embedding for all nodes
    node_ids = G_stellar.nodes()
    node_gen = generator.flow(node_ids) # If we have test train vs sets this 3 lines will be copied
    node_embeddings = model.predict(node_gen)
    
    # Adding extra features
    #np.append(node_embeddings, findArticulationPoints(G))
    #np.append(node_embeddings, calculate_density(G))
    
    return node_embeddings[1]


### Partion Graph with Kernighan-lin

def KernighanLinIterationAndSAGEembedding(minCutEdgeAmount, G):
    
    didItBecomeConnected = False
    graphEmbedding = []
    
    total_vertices = G.number_of_nodes()
    
    # determine total number of iterations
    if total_vertices < 360:
        totalNumberOfIteration = 10
    elif total_vertices > 500:
        totalNumberOfIteration = 0.4 * total_vertices * np.log10(total_vertices)
    else:
        totalNumberOfIteration = total_vertices * np.log10(total_vertices)
    
    
    for j in range(totalNumberOfIteration):
        
        partition = kernighan_lin_bisection(G, max_iter=10)
        
        G_partition1 = G.subgraph(partition[0])
        G_partition2 = G.subgraph(partition[1])
        
        if nx.is_connected(G_partition1) and nx.is_connected(G_partition2):
            
            didItBecomeConnected = True
            
            total_edges = G.number_of_edges()
            partition_1_edges = G_partition1.number_of_edges()
            partition_2_edges = G_partition2.number_of_edges()
            
            edgeBetweenSubGraphs = total_edges-partition_1_edges - partition_2_edges
        
            
            print(str(edgeBetweenSubGraphs) + "        " + str(j))
            
            if edgeBetweenSubGraphs < minCutEdgeAmount: 
                
                # check vertex constraint
                partition_1_vertices = G_partition1.number_of_nodes()
                partition_2_vertices = G_partition2.number_of_nodes()
                
                min_vertex_bound = total_vertices/2 - total_vertices*0.01
                max_vertex_bound = total_vertices/2 + total_vertices*0.01
                
                if ((min_vertex_bound <= partition_1_vertices <= max_vertex_bound)
                        and (min_vertex_bound <= partition_2_vertices <= max_vertex_bound)):
                
                    print('girdi1')   
                    graphEmbeddingTemp = getSAGEembedding(G) 
                    graphEmbedding = np.append(graphEmbeddingTemp, 1)
                    break
            
        # Sona geldiysek ve hala Yes label alamadıysa No label ver
        # Eger hicbir zaman connected bir sekilde bolunemediyse hicbir sey yapma
        if j == totalNumberOfIteration-1 and didItBecomeConnected:
            print('girdi2')
            graphEmbeddingTemp = getSAGEembedding(G) 
            graphEmbedding = np.append(graphEmbeddingTemp, 0)
            break
        
    return didItBecomeConnected, graphEmbedding  