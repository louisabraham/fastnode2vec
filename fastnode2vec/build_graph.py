from pathlib import Path
import pickle


from .graph import Graph


def build_graph(filename, directed, n_edges=None, cache=False):
    filename = Path(filename)
    cached = filename.parent / (filename.name + ".graph.pk")

    if cache and cached.exists():
        print(f"Using cached graph from {cached}.")
        with open(cached, "rb") as f:
            return pickle.load(f)

    def make_edges():
        if filename.name.endswith(".gz"):
            import gzip

            open = gzip.open
        with open(filename, "rt") as f:
            for row in f:
                if row.startswith("#"):
                    continue
                yield row.split()

    graph = Graph(make_edges(), n_edges=n_edges, directed=directed)

    if cache:
        with open(cached, "wb") as f:
            pickle.dump(graph, f)

    return graph


if __name__ == "__main__":

    import click

    @click.command()
    @click.argument("filename", type=click.Path(exists=True))
    @click.option("--n-edges", type=int, default=None)
    def _build_graph(filename, n_edges=None):
        return build_graph(filename, n_edges, cache=True)

    _build_graph()
