# General Libraries
import random as rand
import numpy as np


# Graph Representation & Embedding Library
import networkx as nx


import differentGraphs as dg


from dataEmbeddingSAGE import KernighanLinIterationAndSAGEembedding
from writeToExcel import writeToExcel


### DATA GENERATION


TOTAL_NUMBER_OF_GRAPH_FOR_EACH = 500
NODES_LOW_LIMIT = 60
NODES_HIGH_LIMIT = 200


def dataGenerateAndSave(numberOfNodesLowest, numberOfNodesHighest):
   df=[]
   graphEmbedding = []
   seed = rand.randint(1,1000000)
   numberOfNodes = rand.randint(numberOfNodesLowest,numberOfNodesHighest)
  
   i = TOTAL_NUMBER_OF_GRAPH_FOR_EACH   
   # Number 1 watts_strogatz
   while i>0:
       k_neighbors = rand.randint(2,4)
       probability = rand.random()
       numberOfNodes = rand.randint(numberOfNodesLowest,numberOfNodesHighest)
       G = dg.generate_watts_strogatz_graph(numberOfNodes,seed = seed, k_neighbors=k_neighbors, probability= probability)
      
       isValid = is_graph_appropriate(G)
       if isValid == False:
           continue


       while not nx.is_connected(G):
           k_neighbors = rand.randint(2,4)
           probability = rand.random()
           numberOfNodes = rand.randint(numberOfNodesLowest,numberOfNodesHighest)
           G = dg.generate_watts_strogatz_graph(numberOfNodes,seed = seed, k_neighbors=k_neighbors, probability= probability)
      
       # Iterate to find the best kernighan lin matching
       # TODO buraya matematiği getirilecek
       graphEmbedding = KernighanLinIterationAndSAGEembedding(G)


       if len(graphEmbedding)>0:
           i = i-1
           if len(df) == 0:
               df = graphEmbedding
           else :
               df = np.vstack((df, graphEmbedding))
  
   print('A')
  
   i = TOTAL_NUMBER_OF_GRAPH_FOR_EACH
   # Number 2 Barabasi
   while i>0:
       numberOfNodes = rand.randint(int(numberOfNodesLowest*0.2),int(numberOfNodesHighest*0.6))
       numberOfEdges = rand.randint(1,2)
       G = dg.generate_barabasi_albert_graph(numberOfNodes,seed = seed,edges=numberOfEdges)
       isValid = is_graph_appropriate(G)
       if isValid == False:
           continue


       while not nx.is_connected(G):
           numberOfNodes = rand.randint(int(numberOfNodesLowest*0.2),int(numberOfNodesHighest*0.6))
           numberOfEdges = rand.randint(1,2)
           G = dg.generate_barabasi_albert_graph(numberOfNodes,seed = seed,edges=numberOfEdges)
      
       graphEmbedding = KernighanLinIterationAndSAGEembedding(G)
       if len(graphEmbedding)>0:
           i = i-1
           if len(df) == 0:
               df = graphEmbedding
           else :
               df = np.vstack((df, graphEmbedding))
   print('B')
  
   i = TOTAL_NUMBER_OF_GRAPH_FOR_EACH
   # Number 3 Geometric_graph
   while i>0:
       radius = 0.2
       numberOfNodes = rand.randint(numberOfNodesLowest,numberOfNodesHighest)
       G = dg.generate_random_geometric_graph(numberOfNodes,seed = seed, radius=radius)
       isValid = is_graph_appropriate(G)
       if isValid == False:
           continue


       while not nx.is_connected(G):
           print('C')


           radius = 0.2
           numberOfNodes = rand.randint(numberOfNodesLowest,numberOfNodesHighest)
           G = dg.generate_random_geometric_graph(numberOfNodes,seed = seed, radius=radius)


          
       graphEmbedding = KernighanLinIterationAndSAGEembedding(G)
       if len(graphEmbedding)>0:
           i = i-1
           if len(df) == 0:
               df = graphEmbedding
           else :
               df = np.vstack((df, graphEmbedding))


   print('D')
 
   i = TOTAL_NUMBER_OF_GRAPH_FOR_EACH
   # Number 4 Planar_graph
   while i>0:
       numberOfNodes = rand.randint(numberOfNodesLowest,numberOfNodesHighest)
       numberOfEdges = rand.randint(int(numberOfNodes*1.15),int(numberOfNodes*1.3))
       G = dg.generate_planar_graph(nodes=numberOfNodes,edges=numberOfEdges)
       while not nx.is_connected(G):
           numberOfNodes = rand.randint(numberOfNodesLowest,numberOfNodesHighest)
           numberOfEdges = rand.randint(int(numberOfNodes*1.15),int(numberOfNodes*1.3))
           G = dg.generate_planar_graph(nodes=numberOfNodes,edges=numberOfEdges)


       graphEmbedding = KernighanLinIterationAndSAGEembedding(G)
       if len(graphEmbedding)>0:
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




dataGenerateAndSave(NODES_LOW_LIMIT, NODES_HIGH_LIMIT)

