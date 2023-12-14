import networkx as nx


# Number 1
def generate_watts_strogatz_graph(nodes,seed, k_neighbors = 2, probability = 0.2):
    G = nx.watts_strogatz_graph(nodes, k_neighbors, probability, seed=seed)
    return G

# Number 2
def generate_barabasi_albert_graph(nodes, edges, seed):
    G = nx.barabasi_albert_graph(nodes, edges, seed=seed)
    return G

# Number 3
def generate_erdos_renyi_graph(nodes, probability, seed):
    G = nx.erdos_renyi_graph(nodes, probability, seed=seed)
    return G

# Number 4
def generate_random_geometric_graph(nodes, radius, seed):
    G = nx.random_geometric_graph(nodes, radius, seed=seed)
    return G

# Number 5
# Planer Graph 
def generate_planar_graph(nodes, edges):
    G = nx.gnm_random_graph(n=nodes, m=edges)
    return G

# Number 8
# Tree-like Graph 
def generate_tree_graph(nodes):
    # Create a tree graph with the specified number of nodes and branching factor
    G = nx.generators.trees.random_tree(nodes, 47)
    return G

# Number 9
# Square Grid Graph
def generate_square_grid_graph(rows, columns):
    G = nx.grid_2d_graph(rows, columns)
    return G

# Number 10
# Triangular Grid Graph
def generate_triangular_grid_graph(rows, columns):
    G = nx.triangular_lattice_graph(rows, columns)
    return G

# Number 11
# Complete Graph
def generate_complete_graph(num_nodes):
    G = nx.complete_graph(num_nodes)
    return G
