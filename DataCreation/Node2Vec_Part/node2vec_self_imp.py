
import networkx as nx
from node2vec import Node2Vec


def getEmbedding(G):
    # Precompute probabilities and generate walks
    node2vec = Node2Vec(G, dimensions=16, walk_length=10, num_walks=75, workers=4)

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