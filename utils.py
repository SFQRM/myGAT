import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch as th


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def accuracy(output, labels):
    """
        # A.max(1)：返回A每一行最大值组成的一维数组；
        # [1]则表示返回最大值的索引
    """
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def clustering_coefficient(adjacency):
    adj = adjacency.numpy()
    num_node = adj.shape[0]
    G = nx.from_numpy_matrix(adj, create_using=None)
    G_cluster = nx.clustering(G)
    G_cluster_value = np.array(list(G_cluster.values()))
    G_cluster_value = np.expand_dims(G_cluster_value, 0).repeat(num_node, axis=0)
    G_cluster_value = th.tensor(G_cluster_value, dtype=th.float)
    return G_cluster_value