import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from process_data import load_data
import torch as th

G = nx.Graph()
# G.add_nodes_from([1,2,3,4,5])
# G.add_edges_from([(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)])
# nx.draw(G, node_size=500, with_labels=True)

# plt.show()

path = "./data/cora/"
labels, features, adj, idx_train, idx_val, idx_test = load_data(path=path)

adj = adj.numpy()
# print(np.asmatrix(adj))

G = nx.from_numpy_matrix(adj, create_using=None)
# nx.draw(G, node_size=500, with_labels=True)

# plt.show()

# rint(G)

num_node = 2708
# print(num_node)

G_cluster = nx.clustering(G)
print(type(G_cluster))

value = np.array(list(G_cluster.values()))
value = th.tensor(value)
print(value)
print(type(value))
print(value.shape)


def clustering_coefficient(adjacency):
    adj = adjacency.numpy()
    G = nx.from_numpy_matrix(adj, create_using=None)
    G_cluster = nx.clustering(G)
    G_cluster_value = np.array(list(G_cluster.values()))
    G_cluster_value = th.tensor(G_cluster_value)
    return G_cluster_value
