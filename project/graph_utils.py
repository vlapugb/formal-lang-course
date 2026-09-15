"""Load graph statistics and generate labeled graphs for path queries."""

from pathlib import Path

import cfpq_data
import networkx as nx
import pydot


def get_graph_info(graph_name: str) -> tuple[int, int, set[str]]:
    """Download a dataset graph and return its basic statistics.

    Args:
        graph_name: Name of a graph in the CFPQ_Data dataset.

    Returns:
        A tuple containing the number of vertices, the number of edges,
        and the set of distinct edge labels. Parallel edges and self-loops
        are counted individually.

    Raises:
        FileNotFoundError: The graph name is unknown or its CSV file is missing.
        OSError: Downloaded graph files cannot be written or read.
    """
    graph = cfpq_data.graph_from_csv(cfpq_data.download(graph_name))
    labels = {data["label"] for _, _, data in graph.edges(data=True)}
    return graph.number_of_nodes(), graph.number_of_edges(), labels


def create_two_cycles_graph(
    n: int,
    m: int,
    labels: tuple[str, str],
    output_path: str | Path,
) -> nx.MultiDiGraph:
    """Create two directed cycles with a shared vertex and save them as DOT.

    Args:
        n: Positive number of vertices in the first cycle, excluding the
            shared vertex 0.
        m: Positive number of vertices in the second cycle, excluding the
            shared vertex 0.
        labels: Pair of edge labels, one for each cycle. Labels may coincide.
        output_path: Destination file path. Its parent directory must exist.
            An existing file is overwritten using UTF-8 encoding.

    Returns:
        The generated graph with n + m + 1 vertices and n + m + 2 edges.
        For example, n=2 and m=3 produce cycles of lengths 3 and 4.

    Raises:
        TypeError: A cycle size is not an integer, labels is a string,
            or a label is not a string.
        ValueError: A cycle size is not positive or labels has a length
            other than two.
        OSError: The output file cannot be written.
    """
    if not isinstance(n, int) or not isinstance(m, int):
        raise TypeError("Cycle sizes must be integers")
    if n <= 0 or m <= 0:
        raise ValueError("Cycle sizes must be positive")
    if isinstance(labels, str):
        raise TypeError("Labels must be a pair of strings")
    if len(labels) != 2:
        raise ValueError("Exactly two labels are required")
    if not all(isinstance(label, str) for label in labels):
        raise TypeError("Labels must be strings")

    graph = cfpq_data.labeled_two_cycles_graph(n, m, labels=labels)
    dot = pydot.Dot(graph_type="digraph")
    for node in graph.nodes:
        dot.add_node(pydot.Node(str(node)))
    for source, target, data in graph.edges(data=True):
        dot.add_edge(pydot.Edge(str(source), str(target), label=data["label"]))
    dot.write_raw(str(output_path), encoding="utf-8")
    return graph
