from typing import List, Optional

import numpy as np
from gensim import __version__ as gensim_version
from gensim.models import Word2Vec
from numba import njit
from tqdm.auto import trange

from .graph import Graph


@njit
def set_seed(seed):
    np.random.seed(seed)


class Node2Vec(Word2Vec):
    def __init__(
        self,
        graph: Graph,
        dim: int,
        walk_length: int,
        window: int,
        p: float = 1.0,
        q: float = 1.0,
        workers: int = 1,
        batch_walks: Optional[int] = None,
        use_skipgram: bool = True,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Create a new instance of Node2Vec.

        Parameters
        ---------------------------
        graph: Graph
            Graph object whose nodes are to be embedded.
        dim: int
            The dimensionality of the embedding
        walk_length: int
            Length of the random walk to be computed
        window: int
            The dimension of the window.
            This is HALF of the context, as gensim will
            consider as context all nodes before `window`
            and after `window`.
        p: float = 1.0
            The higher the value, the lower the probability to return to
            the previous node during a walk.
        q: float = 1.0
            The higher the value, the lower the probability to return to
            a node connected to a previous node during a walk.
        workers : int, optional (default = 1)
            The number of threads to use during the embedding process.
            If set to -1, all available threads will be used.
            By setting workers to -1, the Node2Vec model will use all
            available threads during the embedding process.
            This can improve the speed of the embedding process,
            but it can also increase the risk of data-races,
            especially on smaller graphs with a reduced number of nodes.
            It is important to carefully consider the potential
            impact of data-races and whether using all available
            threads is appropriate for a given use case.
            On large graphs, the risk of data-races is generally lower,
            as there are more nodes for the threads to process and
            the likelihood of multiple threads accessing the same
            node at the same time is reduced.
        batch_walks: Optional[int] = None
            Target size (in words) for batches of examples passed to worker threads (and
            thus cython routines).(Larger batches will be passed if individual
            texts are longer than 10000 words, but the standard cython code truncates to that maximum.)
        use_skipgram: bool = True
            Whether to use SkipGram or, alternatively, CBOW as node embedding model.
            PLEASE BE AVISED THAT: In the Node2Vec model, a SkipGram model is used to
            learn low-dimensional node embeddings from graph structure.
            The SkipGram model learns to predict a target node given its surrounding context nodes,
            while the CBOW (Continuous Bag-of-Words) model learns to predict a target
            node given the sum of its context nodes. While it is possible to use a
            CBOW model instead of a SkipGram model in the Node2Vec model,
            it should be noted that this deviates from the original model and may
            not always be the best choice for a given use case.
            The user should carefully consider whether using a CBOW model is
            appropriate for their specific application, as it may perform
            differently than the original SkipGram model.
        seed: Optional[int] = None
            The seed to use to reproduce these experiments.
        **kwargs
            Parameters to be forwarded to parent class.
        """
        # <https://github.com/RaRe-Technologies/gensim/issues/2801>
        assert walk_length < 10000
        if batch_walks is None:
            batch_words = 10000
        else:
            batch_words = min(walk_length * batch_walks, 10000)

        if gensim_version < "4.0.0":
            kwargs["iter"] = 1
            kwargs["size"] = dim
        else:
            kwargs["epochs"] = 1
            kwargs["vector_size"] = dim

        super().__init__(
            sg=int(use_skipgram),
            window=window,
            min_count=1,
            workers=workers,
            batch_words=batch_words,
            **kwargs,
        )
        self.build_vocab(([w] for w in graph.node_names))
        self.graph = graph
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.seed = seed

    def train(self, epochs: int, *, verbose: bool = True, **kwargs):
        """Train the model and compute the node embedding.

        Parameters
        --------------------
        epochs: int
            Number of epochs to train the model for.
        verbose: bool = True
            Whether to show loading bar.
        **kwargs
            Parameters to be forwarded to parent class.
        """

        def gen_nodes():
            """Number of epochs to compute."""
            if self.seed is not None:
                np.random.seed(self.seed)

            for _ in trange(
                epochs,
                dynamic_ncols=True,
                desc="Epochs",
                leave=False,
                disable=not verbose,
            ):
                for i in np.random.permutation(len(self.graph.node_names)):
                    # dummy walk with same length
                    yield [i] * self.walk_length

        super().train(
            gen_nodes(),
            total_examples=epochs * len(self.graph.node_names),
            epochs=1,
            **kwargs,
        )

    def generate_random_walk(self, source_node_id: int) -> List[int]:
        """Returns random walk starting from the provided source node id.

        Parameters
        ----------
        source_node_id: int
            The node ID from which to start the random walk.

        Returns
        ----------
        List containing random walk starting from provided node.
        """
        return self.graph.generate_random_walk(
            self.walk_length, self.p, self.q, source_node_id
        )

    def _do_train_job(self, sentences, alpha, inits):
        """Train the model on a single batch of sentences.

        Parameters
        ----------
        sentences : iterable of list of str
            Corpus chunk to be used in this training batch.
        alpha : float
            The learning rate used in this batch.
        inits : (np.ndarray, np.ndarray)
            Each worker threads private work memory.

        Returns
        -------
        (int, int)
             2-tuple (effective word count after ignoring unknown words and sentence length trimming, total word count).

        """
        if self.seed is not None:
            set_seed(self.seed)
        sentences = [self.generate_random_walk(w[0]) for w in sentences]
        return super()._do_train_job(sentences, alpha, inits)
