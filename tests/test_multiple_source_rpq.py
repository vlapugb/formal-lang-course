"""Compare sparse MS-BFS with independent BFS and the tensor algorithm."""

from collections import deque
import random

import cfpq_data
import networkx as nx
import pytest
from scipy.sparse import csr_matrix

from project.automata_utils import regex_to_dfa
from project.matrix_automata import AdjacencyMatrixFA, tensor_based_rpq
from project.multiple_source_rpq import ms_bfs_based_rpq


def reference_rpq(regex, graph, starts, finals):
    """Explicit BFS over vertex/DFA-state pairs, separately for each source."""
    dfa = regex_to_dfa(regex)
    result = set()
    for source in starts:
        visited = {(source, state) for state in dfa.start_states}
        queue = deque(visited)
        while queue:
            vertex, state = queue.popleft()
            if vertex in finals and state in dfa.final_states:
                result.add((source, vertex))
            for _, target, data in graph.out_edges(vertex, data=True):
                for next_state in dfa(state, data["label"]):
                    pair = (target, next_state)
                    if pair not in visited:
                        visited.add(pair)
                        queue.append(pair)
    return result


def test_multiple_sources_keep_separate_reachability():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([10, 20, 30, 40, -5])
    graph.add_edge(10, 30, label="a")
    graph.add_edge(10, 30, label="a")
    graph.add_edge(20, 30, label="b")
    graph.add_edge(30, 40, label="b")
    graph.add_edge(40, 30, label="a")
    graph.add_edge(40, 40, label="b")
    starts, finals = {10, 20, 30, -5}, {30, 40, -5}
    original = graph.copy()

    assert ms_bfs_based_rpq("a b*", graph, starts, finals) == {(10, 30), (10, 40)}
    assert ms_bfs_based_rpq("epsilon", graph, starts, finals) == {(30, 30), (-5, -5)}
    assert ms_bfs_based_rpq("a*", graph, starts, finals) == {
        (10, 30),
        (30, 30),
        (-5, -5),
    }
    assert ms_bfs_based_rpq("missing", graph, starts, finals) == set()
    assert nx.utils.graphs_equal(graph, original)
    assert starts == {10, 20, 30, -5}
    assert finals == {30, 40, -5}


def test_revisit_vertex_in_different_query_states():
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 10, label="a")
    graph.add_edge(10, 20, label="b")
    assert ms_bfs_based_rpq("a a b", graph, {10}, {20}) == {(10, 20)}


@pytest.mark.parametrize("regex", ["", "epsilon", "a", "a*", "(a | b)*"])
def test_empty_graph(regex):
    assert ms_bfs_based_rpq(regex, nx.MultiDiGraph()) == set()


@pytest.mark.parametrize("boundaries", [(None, None), (set(), set())])
def test_default_boundaries_include_isolated_vertices(boundaries):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([10, 20, 30])
    graph.add_edge(10, 20, label="token")
    assert ms_bfs_based_rpq("epsilon", graph, *boundaries) == {
        (10, 10),
        (20, 20),
        (30, 30),
    }
    assert ms_bfs_based_rpq("token", graph, *boundaries) == {(10, 20)}


@pytest.mark.parametrize("starts, finals", [({99}, {10}), ({10}, {99})])
def test_unknown_boundary_vertices(starts, finals):
    graph = nx.MultiDiGraph()
    graph.add_node(10)
    with pytest.raises(ValueError, match="graph vertices"):
        ms_bfs_based_rpq("epsilon", graph, starts, finals)


@pytest.mark.parametrize("seed", range(8))
def test_matches_reference_and_tensor_on_random_graphs(seed):
    rng = random.Random(seed)
    nodes = [-10, 20, 40, 100, 200]
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(nodes + [999])
    for _ in range(20):
        graph.add_edge(
            rng.choice(nodes), rng.choice(nodes), label=rng.choice(["a", "b"])
        )
    starts, finals = {-10, 40, 999}, {20, 100, 999}
    queries = ["", "epsilon", "a", "a*", "a b", "(a | b)*", "(a b)*", "a* b a*"]
    for regex in queries:
        expected = reference_rpq(regex, graph, starts, finals)
        assert ms_bfs_based_rpq(regex, graph, starts, finals) == expected
        assert tensor_based_rpq(regex, graph, starts, finals) == expected


def test_long_chain_uses_no_dense_matrices_or_closure(monkeypatch):
    graph = nx.MultiDiGraph()
    for vertex in range(130):
        graph.add_edge(vertex, vertex + 1, label="a")

    def forbidden(*args, **kwargs):
        raise AssertionError("MS-BFS must not use dense matrices or a closure")

    monkeypatch.setattr(AdjacencyMatrixFA, "transitive_closure", forbidden)
    monkeypatch.setattr(csr_matrix, "toarray", forbidden)
    monkeypatch.setattr(csr_matrix, "todense", forbidden)
    assert ms_bfs_based_rpq("a*", graph, {0, 65, 130}, {130}) == {
        (0, 130),
        (65, 130),
        (130, 130),
    }


@pytest.fixture(
    scope="module",
    params=[name for name in cfpq_data.DATASET if name in {"skos", "generations"}],
)
def dataset_graph(request):
    """Download each small real dataset graph once for all query checks."""
    return cfpq_data.graph_from_csv(cfpq_data.download(request.param))


def test_real_dataset_graphs(dataset_graph):
    graph = dataset_graph
    nodes = sorted(graph.nodes)
    labels = sorted(nx.get_edge_attributes(graph, "label").values())
    first, last = labels[0], labels[-1]
    starts, finals = set(nodes[::13]), set(nodes[::7])
    queries = ["epsilon", first, f"{first}*", f"({first} | {last})*", f"{first} {last}"]
    for regex in queries:
        assert ms_bfs_based_rpq(regex, graph, starts, finals) == reference_rpq(
            regex, graph, starts, finals
        )


def test_single_label_query_on_dataset_has_exact_edge_pairs(dataset_graph):
    graph = dataset_graph
    label = min(nx.get_edge_attributes(graph, "label").values())
    expected = {
        (source, target)
        for source, target, data in graph.edges(data=True)
        if data["label"] == label
    }
    assert ms_bfs_based_rpq(label, graph) == expected
