from collections import defaultdict

import numpy as np

from numba import njit
from scipy.sparse import csr_matrix
from tqdm import tqdm

@njit(nogil=True)
def _neighbors(indptr, indices, t):
    return indices[indptr[t] : indptr[t + 1]]

@njit(nogil=True)
def _random_walk(indptr, indices, walk_length, p, q, t):
    max_prob = max(1 / p, 1, 1 / q)
    prob_0 = 1 / p / max_prob
    prob_1 = 1 / max_prob
    prob_2 = 1 / q / max_prob

    walk = np.empty(walk_length, dtype=indices.dtype)
    walk[0] = t
    walk[1] = np.random.choice(_neighbors(indptr, indices, t))
    for j in range(2, walk_length):
        neighbors = _neighbors(indptr, indices, walk[j - 1])
        if p == q == 1:
            # faster version
            walk[j] = np.random.choice(neighbors)
            continue
        while True:
            new_node = np.random.choice(neighbors)
            r = np.random.rand()
            if new_node == walk[j - 2]:
                if r < prob_0:
                    break
            elif np.searchsorted(_neighbors(indptr, indices, walk[j - 2]), new_node):
                if r < prob_1:
                    break
            elif r < prob_2:
                break
        walk[j] = new_node
    return walk


class Graph:
    def __init__(self, edges, directed, n_edges=None):
        if n_edges is None:
            try:
                n_edges = len(edges)
            except TypeError:
                pass

        nodes = defaultdict(lambda: len(nodes))
        from_ = []
        to = []
        for a, b in tqdm(edges, desc="Reading graph", total=n_edges):
            a = nodes[a]
            b = nodes[b]
            from_.append(a)
            to.append(b)
            if not directed:
                from_.append(b)
                to.append(a)

        n = len(nodes)

        edges = csr_matrix((np.ones(len(from_), dtype=bool), (from_, to)), shape=(n, n))
        edges.sort_indices()
        self.indptr = edges.indptr
        self.indices = edges.indices

        node_names = [None] * n
        for name, i in nodes.items():
            node_names[i] = name
        self.node_names = np.array(node_names)

    def generate_random_walk(self, walk_length, p, q, start):
        walk = _random_walk(self.indptr, self.indices, walk_length, p, q, start)
        return self.node_names[walk].tolist()
