"""Check matrix-based CFPQ against the Hellings semantics."""

import random
from copy import deepcopy

import networkx as nx
import pytest
from pyformlang.cfg import CFG, Variable

from project.hellings import hellings_based_cfpq
from project.matrix_cfpq import matrix_based_cfpq


def test_nested_language_with_custom_start():
    grammar = CFG.from_text("Q -> a Q b | epsilon", start_symbol=Variable("Q"))
    graph = nx.DiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 30, label="b")
    assert matrix_based_cfpq(grammar, graph, {10}, {30}) == {(10, 30)}


def test_epsilon_only_includes_isolated_vertices():
    graph = nx.DiGraph()
    graph.add_nodes_from([10, 99])
    assert matrix_based_cfpq(CFG.from_text("S -> epsilon"), graph) == {
        (10, 10),
        (99, 99),
    }


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiDiGraph])
def test_empty_graph_and_empty_language(graph_type):
    assert matrix_based_cfpq(CFG.from_text("S -> epsilon"), graph_type()) == set()
    graph = graph_type()
    graph.add_edge(10, 20, label="a")
    assert matrix_based_cfpq(CFG.from_text("S -> A\nA -> S"), graph) == set()
    assert matrix_based_cfpq(CFG(start_symbol=Variable("S")), graph) == set()


def test_parallel_edges_and_noncontiguous_identifiers():
    graph = nx.MultiDiGraph()
    graph.add_edge(-5, 40, label="a")
    graph.add_edge(-5, 40, label="b")
    graph.add_edge(-5, 40, label="a")
    assert matrix_based_cfpq(CFG.from_text("S -> a | b"), graph) == {(-5, 40)}


@pytest.mark.parametrize(
    "starts, finals, expected",
    [
        (None, None, {(10, 20), (30, 20)}),
        (set(), set(), {(10, 20), (30, 20)}),
        ({10}, None, {(10, 20)}),
        ({10}, set(), {(10, 20)}),
        (None, {20}, {(10, 20), (30, 20)}),
        (set(), {30}, set()),
    ],
)
def test_boundary_conventions(starts, finals, expected):
    graph = nx.DiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(30, 20, label="a")
    graph.add_node(99)
    assert matrix_based_cfpq(CFG.from_text("S -> a"), graph, starts, finals) == expected


@pytest.mark.parametrize("starts, finals", [({99}, None), (None, {99})])
def test_unknown_boundary_vertices(starts, finals):
    graph = nx.DiGraph()
    graph.add_node(10)
    with pytest.raises(ValueError, match="graph vertices"):
        matrix_based_cfpq(CFG.from_text("S -> epsilon"), graph, starts, finals)


def test_boundaries_do_not_prune_intermediate_vertices():
    graph = nx.DiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 30, label="b")
    assert matrix_based_cfpq(CFG.from_text("S -> a b"), graph, {10}, {30}) == {(10, 30)}


def test_repeated_queries_preserve_all_inputs():
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, label="a", metadata={"weight": 2})
    graph.add_edge(20, 30, label="b")
    graph.add_node(99, description="isolated")
    original_graph = deepcopy(graph)
    grammar = CFG.from_text("S -> a A | epsilon\nA -> b")
    grammar_snapshot = (
        set(grammar.productions),
        set(grammar.variables),
        set(grammar.terminals),
        grammar.start_symbol,
    )
    starts, finals = {10, 99}, {30, 99}
    for _ in range(2):
        assert matrix_based_cfpq(grammar, graph, starts, finals) == {
            (10, 30),
            (99, 99),
        }
        assert nx.utils.graphs_equal(graph, original_graph)
        assert starts == {10, 99}
        assert finals == {30, 99}
        assert (
            set(grammar.productions),
            set(grammar.variables),
            set(grammar.terminals),
            grammar.start_symbol,
        ) == grammar_snapshot


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    "text",
    [
        "S -> a",
        "S -> a S b | epsilon",
        "S -> S S | a | epsilon",
        "S -> A B\nA -> a A | epsilon\nB -> b B | epsilon",
    ],
)
def test_matches_hellings_on_random_graphs(text, seed):
    rng = random.Random(seed)
    vertices = [10, 30, 50]
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(vertices + [99])
    for _ in range(12):
        graph.add_edge(
            rng.choice(vertices), rng.choice(vertices), label=rng.choice("ab")
        )
    starts = set(rng.sample(vertices + [99], rng.randint(1, len(vertices) + 1)))
    finals = set(rng.sample(vertices + [99], rng.randint(1, len(vertices) + 1)))
    grammar = CFG.from_text(text)
    assert matrix_based_cfpq(
        deepcopy(grammar), deepcopy(graph), starts, finals
    ) == hellings_based_cfpq(deepcopy(grammar), deepcopy(graph), starts, finals)
