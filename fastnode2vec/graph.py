from collections import defaultdict

import numpy as np
from typing import List, Iterable, Optional
from numba import njit
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm


@njit(nogil=True)
def _csr_row_cumsum(indptr, data):
    out = np.empty_like(data)
    for i in range(len(indptr) - 1):
        acc = 0
        for j in range(indptr[i], indptr[i + 1]):
            acc += data[j]
            out[j] = acc
        out[j] = 1.0
    return out


@njit(nogil=True)
def _neighbors(indptr, indices_or_data, t):
    return indices_or_data[indptr[t] : indptr[t + 1]]


@njit(nogil=True)
def _isin_sorted(a, x):
    return a[np.searchsorted(a, x)] == x


@njit(nogil=True)
def _random_walk(indptr, indices, walk_length, p, q, t):
    max_prob = max(1 / p, 1, 1 / q)
    prob_0 = 1 / p / max_prob
    prob_1 = 1 / max_prob
    prob_2 = 1 / q / max_prob

    walk = np.empty(walk_length, dtype=indices.dtype)
    walk[0] = t
    neighbors = _neighbors(indptr, indices, t)
    if not neighbors.size:
        return walk[:1]
    walk[1] = np.random.choice(_neighbors(indptr, indices, t))
    for j in range(2, walk_length):
        neighbors = _neighbors(indptr, indices, walk[j - 1])
        if not neighbors.size:
            return walk[:j]
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
            elif _isin_sorted(_neighbors(indptr, indices, walk[j - 2]), new_node):
                if r < prob_1:
                    break
            elif r < prob_2:
                break
        walk[j] = new_node
    return walk


@njit(nogil=True)
def _random_walk_weighted(indptr, indices, data, walk_length, p, q, t):
    max_prob = max(1 / p, 1, 1 / q)
    prob_0 = 1 / p / max_prob
    prob_1 = 1 / max_prob
    prob_2 = 1 / q / max_prob

    walk = np.empty(walk_length, dtype=indices.dtype)
    walk[0] = t
    neighbors = _neighbors(indptr, indices, t)
    if not neighbors.size:
        return walk[:1]
    walk[1] = _neighbors(indptr, indices, t)[
        np.searchsorted(_neighbors(indptr, data, t), np.random.rand())
    ]
    for j in range(2, walk_length):
        neighbors = _neighbors(indptr, indices, walk[j - 1])
        if not neighbors.size:
            return walk[:j]
        neighbors_p = _neighbors(indptr, data, walk[j - 1])
        if p == q == 1:
            # faster version
            walk[j] = neighbors[np.searchsorted(neighbors_p, np.random.rand())]
            continue
        while True:
            new_node = neighbors[np.searchsorted(neighbors_p, np.random.rand())]
            r = np.random.rand()
            if new_node == walk[j - 2]:
                if r < prob_0:
                    break
            elif _isin_sorted(_neighbors(indptr, indices, walk[j - 2]), new_node):
                if r < prob_1:
                    break
            elif r < prob_2:
                break
        walk[j] = new_node
    return walk


class Graph:
    """Graph object to be used for sampling Node2Vec-style walks."""

    def __init__(self, edges: Iterable, directed: bool, weighted: bool, verbose: bool = True, number_of_edges: Optional[int]=None):
        """Create new Graph object.
        
        Parameters
        ------------
        edges: Iterable
            Iteratable of tuples with the edges of the graph.
            Note that we expect tuples of the form (src, dst)
            in the case of an unweighted graph, and triples
            of the form (src, dst, weight) in the case of a directed graph.
        directed: bool
            Whether to load the graph as directed or undirected.
        weighted: bool
            Whether to expect the graph has weights.
        verbose: bool = True
            Whether to show a loading bar.
        number_of_edges: Optional[int] = None
            The expected number of edges to to shown in the loading bar.
        """
        if number_of_edges is None:
            try:
                number_of_edges = len(edges)
            except TypeError:
                pass

        self.weighted = weighted

        nodes = defaultdict(lambda: len(nodes))

        from_ = []
        to = []
        if weighted:
            data = []
        for tpl in tqdm(
            edges,
            desc="Reading graph",
            dynamic_ncols=True,
            leave=False,
            disable=not verbose,
            total=number_of_edges
        ):
            if weighted:
                a, b, w = tpl
                data.append(w)
            else:
                a, b = tpl
            a = nodes[a]
            b = nodes[b]
            from_.append(a)
            to.append(b)
            if not directed:
                from_.append(b)
                to.append(a)
                if weighted:
                    data.append(w)

        if not weighted:
            data = np.ones(len(from_), dtype=bool)

        n = len(nodes)

        edges = csr_matrix((data, (from_, to)), shape=(n, n))
        edges.sort_indices()
        self.indptr = edges.indptr
        self.indices = edges.indices
        if weighted:
            data = edges.data / edges.sum(axis=1).A1.repeat(np.diff(self.indptr))
            self.data = _csr_row_cumsum(self.indptr, data)

        node_names = [None] * n
        for name, i in nodes.items():
            node_names[i] = name
        self.node_names = np.array(node_names)

    def generate_random_walk(self, walk_length: int, p: float, q: float, source_node_id: int) -> List[int]:
        """Returns random walk starting from the provided source node id.

        Parameters
        ----------
        walk_length: int
            Length of the random walk to be computed
        p: float = 1.0
            The higher the value, the lower the probability to return to
            the previous node during a walk.
        q: float = 1.0
            The higher the value, the lower the probability to return to
            a node connected to a previous node during a walk.
        source_node_id: int
            The node ID from which to start the random walk.

        Returns
        ----------
        List containing random walk starting from provided node.
        """
        if self.weighted:
            walk = _random_walk_weighted(
                self.indptr, self.indices, self.data, walk_length, p, q, source_node_id
            )
        else:
            walk = _random_walk(self.indptr, self.indices, walk_length, p, q, source_node_id)
        return self.node_names[walk].tolist()
