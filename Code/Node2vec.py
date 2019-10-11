import numpy as np
import networkx as nx
from gensim.models import Word2Vec

import warnings
warnings.filterwarnings("ignore")


edge_functions = {
    "hadamard": lambda a, b: a * b,
    "average": lambda a, b: 0.5 * (a + b),
    "l1": lambda a, b: np.abs(a - b),
    "l2": lambda a, b: np.abs(a - b) ** 2,
}

class N2V:
    def __init__(self, p, q, num_walks, walk_length, dimensions, workers):
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.workers = workers
        self.wvecs = None

    def edges_to_features(self, edge_list, edge_function, dimensions):
        n_tot = len(edge_list)
        feature_vec = np.empty((n_tot, dimensions), dtype='f')
        for ii in range(n_tot):
            v1, v2 = edge_list[ii]
            emb1 = np.asarray(self.wvecs[str(v1)])
            emb2 = np.asarray(self.wvecs[str(v2)])
            feature_vec[ii] = edge_function(emb1, emb2)
        return feature_vec

    def get_alias_edges(self, g, src, dest, p=1, q=1):
        probs = []
        for nei in sorted(g.neighbors(dest)):
            if nei == src:
                probs.append(1 / p)
            elif g.has_edge(nei, src):
                probs.append(1)
            else:
                probs.append(1 / q)
        norm_probs = [float(prob) / sum(probs) for prob in probs]
        return self.get_alias_nodes(norm_probs)

    def get_alias_nodes(self, probs):
        l = len(probs)
        a, b = np.zeros(l), np.zeros(l, dtype=np.int)
        small, large = [], []

        for i, prob in enumerate(probs):
            a[i] = l * prob
            if a[i] < 1.0:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            sma, lar = small.pop(), large.pop()
            b[sma] = lar
            a[lar] += a[sma] - 1.0
            if a[lar] < 1.0:
                small.append(lar)
            else:
                large.append(lar)
        return b, a


    def preprocess_transition_probs(self, g, directed=False, p=1, q=1):
        alias_nodes, alias_edges = {}, {}
        for node in g.nodes():
            probs = [g[node][nei]['weight'] for nei in sorted(g.neighbors(node))]
            norm_const = sum(probs)
            norm_probs = [float(prob) / norm_const for prob in probs]
            alias_nodes[node] = self.get_alias_nodes(norm_probs)

        if directed:
            for edge in g.edges():
                alias_edges[edge] = self.get_alias_edges(g, edge[0], edge[1], p, q)
                # print(alias_edges[edge])
        else:
            for edge in g.edges():
                alias_edges[edge] = self.get_alias_edges(g, edge[0], edge[1], p, q)
                alias_edges[(edge[1], edge[0])] = self.get_alias_edges(g, edge[1], edge[0], p, q)

        return alias_nodes, alias_edges

    def node2vec_walk(self, g, start, alias_nodes, alias_edges, walk_length = 50):
        path = [start]
        walk_length = self.walk_length
        while len(path) < walk_length:
            node = path[-1]
            neis = sorted(g.neighbors(node))
            if len(neis) > 0:
                if len(path) == 1:
                    l = len(alias_nodes[node][0])
                    idx = int(np.floor(np.random.rand() * l))
                    if np.random.rand() < alias_nodes[node][1][idx]:
                        path.append(neis[idx])
                    else:
                        path.append(neis[alias_nodes[node][0][idx]])
                else:
                    prev = path[-2]
                    l = len(alias_edges[(prev, node)][0])
                    idx = int(np.floor(np.random.rand() * l))
                    if np.random.rand() < alias_edges[(prev, node)][1][idx]:
                        path.append(neis[idx])
                    else:
                        path.append(neis[alias_edges[(prev, node)][0][idx]])
            else:
                break
        return path

    def learn_embeddings(walks, dimensions, workers = 8, window_size=10, niter=5):
        '''
        Learn embeddings by optimizing the Skipgram objective using SGD.
        '''
        # TODO: Python27 only
        # walks = [map(str, walk) for walk in walks]
        model = Word2Vec(walks,
                         size=dimensions,
                         window=window_size,
                         min_count=0,
                         sg=1,
                         workers= workers,
                         iter=niter)
        return model.wv
