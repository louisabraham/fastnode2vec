import logging

import click

from .build_graph import build_graph
from .node2vec import Node2Vec


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("--directed", is_flag=True)
@click.option("--dim", type=int, required=True)
@click.option("--p", type=float, default=1)
@click.option("--q", type=float, default=1)
@click.option("--walk-length", type=int, required=True)
@click.option("--context", type=int, default=10)
@click.option("--epochs", type=int, required=True)
@click.option("--workers", type=int, default=1)
@click.option("--batch-walks", type=int, default=None)
@click.option("--debug", type=click.Path(), default=None)
@click.option("--output", type=click.Path(), default=None)
def node2vec(
    filename: str,
    directed: bool,
    dim,
    p: float,
    q: float,
    walk_length: int,
    context: int,
    epochs: int,
    workers: int,
    batch_walks: int,
    debug: str or None,
    output: str or None,
):

    if debug is not None:
        logging.basicConfig(
            format="%(asctime)s : %(levelname)s : %(message)s",
            level=logging.DEBUG,
            filename=debug,
        )

    graph = build_graph(filename, directed)
    n2v = Node2Vec(
        graph,
        dim,
        walk_length,
        context=context,
        p=p,
        q=q,
        workers=workers,
        batch_walks=batch_walks,
    )
    print("Graph loaded")

    n2v.train(epochs)

    if output is None:
        output = filename + ".wv"
    n2v.wv.save(output)


if __name__ == "__main__":
    node2vec()
