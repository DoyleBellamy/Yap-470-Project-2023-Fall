import networkx as nx
import matplotlib.pyplot as plt

def create_connected_watts_strogatz_graph(n=100,k=3,p=0.2):
    G = nx.connected_watts_strogatz_graph(n, k, p)
    #nx.draw(G,with_labels= True, font_weight = 'bold')
    #plt.show()
    return G
