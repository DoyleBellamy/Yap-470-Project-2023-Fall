import networkx as nx
import matplotlib.pyplot as plt

def create_connected_erdos_renyi_graph(n=100,p=0.2):
    G = nx.fast_gnp_random_graph(n,p)
    #nx.draw(G,with_labels= True, font_weight = 'bold')
    #plt.show()
    while not nx.is_connected(G):
        G = nx.fast_gnp_random_graph(n, p)
    return G
