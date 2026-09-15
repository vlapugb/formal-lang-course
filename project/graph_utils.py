"""Load graph statistics and generate labeled graphs for path queries."""

from pathlib import Path

import cfpq_data
import networkx as nx
import pydot





def get_graph_info(graph_name: str) -> tuple[int, int, set[str]]:
    graph = cfpq_data.graph_from_csv(cfpq_data.download(graph_name))
    labels = {data["label"] for _, _, data in graph.edges(data=True)}
    return graph.number_of_nodes(), graph.number_of_edges(), labels


def create_two_cycles_graph(
    n: int,
    m: int,
    labels: tuple[str, str],
    output_path: str | Path,
) -> nx.MultiDiGraph:
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
