from gensim.models import Word2Vec
import numpy as np
from numba import njit

from tqdm import tqdm


@njit
def set_seed(seed):
    np.random.seed(seed)


class Node2Vec(Word2Vec):
    def __init__(
        self,
        graph,
        dim,
        walk_length,
        context,
        p=1.0,
        q=1.0,
        workers=1,
        batch_walks=None,
        seed=None,
        **args,
    ):
        # <https://github.com/RaRe-Technologies/gensim/issues/2801>
        assert walk_length < 10000
        if batch_walks is None:
            batch_words = 10000
        else:
            batch_words = min(walk_length * batch_walks, 10000)

        super().__init__(
            sg=1,
            iter=1,
            size=dim,
            window=context,
            min_count=1,
            workers=workers,
            batch_words=batch_words,
            **args,
        )
        self.build_vocab(([w] for w in graph.node_names))
        self.graph = graph
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.seed = seed

    def train(self, epochs, *, progress_bar=True, **kwargs):
        def gen_nodes(epochs):
            if self.seed is not None:
                np.random.seed(self.seed)
            for _ in range(epochs):
                for i in np.random.permutation(len(self.graph.node_names)):
                    # dummy walk with same length
                    yield [i] * self.walk_length

        if progress_bar:

            def pbar(it):
                return tqdm(
                    it, desc="Training", total=epochs * len(self.graph.node_names)
                )

        else:

            def pbar(it):
                return it

        super().train(
            pbar(gen_nodes(epochs)),
            total_examples=epochs * len(self.graph.node_names),
            epochs=1,
            **kwargs,
        )

    def generate_random_walk(self, t):
        return self.graph.generate_random_walk(self.walk_length, self.p, self.q, t)

    def _do_train_job(self, sentences, alpha, inits):
        if self.seed is not None:
            set_seed(self.seed)
        sentences = [self.generate_random_walk(w[0]) for w in sentences]
        return super()._do_train_job(sentences, alpha, inits)
