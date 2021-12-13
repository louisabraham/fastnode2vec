[![Downloads](https://pepy.tech/badge/fastnode2vec)](https://pepy.tech/project/fastnode2vec) [![PyPI
version](https://badge.fury.io/py/fastnode2vec.svg)](https://badge.fury.io/py/fastnode2vec)
[![DOI](https://zenodo.org/badge/257390910.svg)](https://zenodo.org/badge/latestdoi/257390910)


# fastnode2vec

*Really* fast implementation of node2vec based on [numba](https://numba.pydata.org/) and [gensim](https://radimrehurek.com/gensim/). Memory usage is **linear** and scales with your data unlike most other implementations. The algorithm is described in this [blog post](https://louisabraham.github.io/articles/node2vec-sampling.html).

## API

`Node2Vec` inherits from gensim's [`Word2Vec`](https://radimrehurek.com/gensim/models/word2vec.html), all its APi is valid.

```python
from fastnode2vec import Graph, Node2Vec

graph = Graph([("a", "b"), ("b", "c"), ("c", "a"), ("a", "d")],
              directed=False, weighted=False)

# or
graph = Graph([("a", "b", 1), ("b", "c", 2), ("c", "a", 3), ("a", "d", 4)],
              directed=False, weighted=True)

n2v = Node2Vec(graph, dim=10, walk_length=100, context=10, p=2.0, q=0.5, workers=2)

n2v.train(epochs=100)

print(n2v.wv["a"])
```

## CLI


```
Usage: fastnode2vec [OPTIONS] FILENAME

Options:
  --directed
  --weighted
  --dim INTEGER          [required]
  --p FLOAT
  --q FLOAT
  --walk-length INTEGER  [required]
  --context INTEGER
  --epochs INTEGER       [required]
  --workers INTEGER
  --batch-walks INTEGER
  --debug PATH
  --output PATH
  --help                 Show this message and exit.

```


Compute embeddings of the [Gnutella peer-to-peer network](https://snap.stanford.edu/data/p2p-Gnutella08.html):

```
wget https://snap.stanford.edu/data/p2p-Gnutella08.txt.gz
fastnode2vec p2p-Gnutella08.txt.gz --dim 16 --walk-length 100 --epochs 10 --workers 2
```

## Load embeddings produced by the CLI

Just use the [`Word2Vec`](https://radimrehurek.com/gensim/models/word2vec.html) API.

```python
from gensim.models import KeyedVectors

wv = KeyedVectors.load("p2p-Gnutella08.txt.gz.wv", mmap='r')
```

## Citing

If you have used this software in a scientific publication, please cite it using the following BibLaTeX code:

```
@software{fastnode2vec,
  author       = {Louis Abraham},
  title        = {fastnode2vec},
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3902632},
  url          = {https://doi.org/10.5281/zenodo.3902632}
}
```
