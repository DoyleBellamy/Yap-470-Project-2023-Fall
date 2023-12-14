import networkx as nx
import random as rand
import matplotlib.pyplot as plt
import numpy as np
from kernighan_lin import kernighan_lin_bisection

from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from tensorflow.keras import layers, optimizers, losses, Model
from stellargraph.layer import MeanPoolingAggregator

from test import get_degree_feature
from test import get_betweenness_centrality_feature
from test import generate_node_features

import differentGraphs as dg

# We have 12 different types of graph
# Get 100 from each one of them
# Total of 1200 different graph embedding we will get (With different node numbers)
# Then pass them to dataGeneratePart

# TODO edges between partition'a bakılacak onun formülü eklenecek
# TODO Embedding'in sonuna bizim custom koyacaklarimiz eklenecek

TOTAL_NUMBER_OF_GRAPH_FOR_EACH = 10
NODES_LOW_LIMIT = 60
NODES_HIGH_LIMIT = 200

excel_file_path = 'output.xlsx'

def writeToExcel(embeddings):
    df = pd.DataFrame(embeddings, columns=[f'Column{i}' for i in range(1, len(embeddings[0])+1)])
    if os.path.exists(excel_file_path):
        # İkinci DataFrame'i dosyaya ekleyerek yaz
        with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a') as writer:
            df.to_excel(writer, index=False, header=False)
    else:
        df.to_excel(excel_file_path, index=False)

def generateData():
    df=[]
    print(len(df))
    i = 0
    for i in range(4):
        # Burada olabilecek tum graph cesitlerini koyup ayri ayri deneyecegiz"
        graphEmbedding = yesLabeledEmbedding(totalNumberOfNodes= 100, edgesBetweenPartitions=3)
        if i == 0:
            df = graphEmbedding
        else : 
            df = np.vstack((df, graphEmbedding))
    writeToExcel(df)

def get_embedding_SAGE(G):
    
    feature_functions = {
        'degree': get_degree_feature,
        'betweenness_centrality': get_betweenness_centrality_feature,
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
    
    return node_embeddings[1]


def dictionaryToNpArray(embedding):
    array_from_dict = np.array(list(embedding.values()))
    return array_from_dict

def KernighanLinIterationAndSAGEembedding(totalNumberOfIteration, G):
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
                nodeEmbeddings = get_embedding_SAGE(G) 
                nodeEmbeddingsArray = dictionaryToNpArray(nodeEmbeddings)
                graphEmbedding = np.mean(nodeEmbeddingsArray, axis=0)
                graphEmbedding = np.append(graphEmbedding, 1)
                break
            
        # Sona geldiysek ve hala Yes label alamadıysa No label ver
        # Eger hicbir zaman connected bir sekilde bolunemediyse hicbir sey yapma
        if j == totalNumberOfIteration-1 and didItBecomeConnected:
            print('girdi2')
            nodeEmbeddings = get_embedding_SAGE(G) 
            nodeEmbeddingsArray = dictionaryToNpArray(nodeEmbeddings)
            graphEmbedding = np.mean(nodeEmbeddingsArray, axis=0)
            graphEmbedding = np.append(graphEmbedding, 0)
            break
    return didItBecomeConnected,graphEmbedding    

def dataGenerateAndSave(numberOfNodesLowest, numberOfNodesHighest):
    df=[]
    graphEmbedding = []
    seed = rand.randint(1,1000000)
    numberOfNodes = rand.randint(numberOfNodesLowest,numberOfNodesHighest)
    
    
    '''
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
    '''
    '''
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
    '''
    '''
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

    '''

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
        didItBecomeConnected, graphEmbedding = KernighanLinIterationAndSAGEembedding(totalNumberOfIteration,G)
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



dataGenerateAndSave(NODES_LOW_LIMIT,NODES_HIGH_LIMIT)
