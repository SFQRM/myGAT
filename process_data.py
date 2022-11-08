import torch
import numpy as np
import scipy.sparse as sp
from utils import encode_onehot, normalize_features, normalize_adj


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    # 读取.content文件并转化为字符串
    """
        np.genfromtxt(): 从文件读取数据，并转化成为指定类型。
    """
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))

    """获取label"""
    # 取最后一列的label并将label用one-hot编码
    labels = encode_onehot(idx_features_labels[:, -1])
    # 将one-hot编码的label转化成数值化的label
    labels = torch.LongTensor(np.where(labels)[1])
    # print(labels)       # tensor([2, 5, 4,  ..., 1, 0, 2])

    """获取特征"""
    # 取第一列到倒数第二列为特征
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # print(features.shape)       # (2708, 1433)
    # 规范化特征
    features = normalize_features(features)
    features = torch.FloatTensor(np.array(features.todense()))

    """build graph"""
    # 取第一列的paper-id
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # 将paper-id与序号对应
    idx_map = {j: i for i, j in enumerate(idx)}
    # print(idx_map)      # {31336: 0, 1061127: 1, 1106406: 2, 13195: 3, ..., 24043: 2707}

    # 读取.cites文件并转化为整型list
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    # 将.cites文件中的paper-id之间的引用关系映射成idx_map中的序号引用
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    # print(edges)

    """获取邻接矩阵"""
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print(adj.shape)          # (2708, 2708)
    # 规范化邻接矩阵
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    # print(adj)
    adj = torch.FloatTensor(np.array(adj.todense()))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return labels, features, adj, idx_train, idx_val, idx_test