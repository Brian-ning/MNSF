import numpy as np
from sklearn.manifold import TSNE
from sklearn.utils import Bunch
import openne.node2vec as node2vec
import networkx as nx
from sklearn.metrics import roc_auc_score
from ..Network_Fusion.fusion import snf_wrapper as sw


def cluster(nx_graph, ground_truth, Parameter, K1, mu, K2, t, tsne=False):
    wvecs = []
    for G in nx_graph:
        model_nf = node2vec.Node2vec(G, Parameter["walk_length"], Parameter["num_walks"], Parameter["dimensions"], p=1, q=1, dw=True)
        index_num = sorted([int(i) for i in model_nf.vectors.keys()])
        g_embedding = [model_nf.vectors[str(i)] for i in index_num]
        wvecs.append(np.array(g_embedding))
    snf_wrapper = sw.snf_wrapper(wvecs, ground_truth)
    network, ground_truth, best, second = snf_wrapper.fusion(K1, mu, K2, t)
    return network, ground_truth, best, second

def cluster_E(nx_graph, ground_truth, Parameter, nodes, K1, mu, K2, t, tsne=False):
    # 基于node2vec的表示学习过程
    wvecs = []
    arr_size = len(nodes)
    walk_length = Parameter["walk_length"]
    num_walks = Parameter["num_walks"]
    dimensions = Parameter["dimensions"]
    p = Parameter["p"]
    q = Parameter["q"]
    each_data = np.random.random((arr_size, dimensions))
    for G in nx_graph:
        model_nf = node2vec.Node2vec(G, walk_length, num_walks, dimensions, p=p, q =q, dw=True)
        index_num = sorted([int(i) for i in model_nf.vectors.keys()])
        np.array([model_nf.vectors[str(i)] for i in index_num])
        for i in index_num:
            each_data[nodes.index(i)] = np.array(model_nf.vectors[str(i)])
        wvecs.append(np.array(each_data))
        # g_embedding = [model_nf.vectors[str(i)] for i in index_num]
        # wvecs.append(np.array(g_embedding))

    # SNF算法过程
    snf_wrapper = sw.snf_wrapper(wvecs, ground_truth)
    network, ground_truth, best, second = snf_wrapper.fusion_E(K1, mu, K2, t)

    # 返回学习到的网络
    return network, ground_truth, best, second
