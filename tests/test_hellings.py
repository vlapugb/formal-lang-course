"""Check context-free endpoint relations and the existing graph conventions."""

import random
from copy import deepcopy

import networkx as nx
import pytest
from pyformlang.cfg import CFG, Production, Terminal, Variable

from project.hellings import hellings_based_cfpq
from project.multiple_source_rpq import ms_bfs_based_rpq


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiDiGraph])
def test_nested_language_and_custom_start(graph_type):
    grammar = CFG.from_text("Q -> a Q b | epsilon", start_symbol=Variable("Q"))
    graph = graph_type()
    graph.add_nodes_from([10, 20, 30, 40, 50, 99])
    for source, target, label in [
        (10, 20, "a"),
        (20, 30, "a"),
        (30, 40, "b"),
        (40, 50, "b"),
    ]:
        graph.add_edge(source, target, label=label)
    assert hellings_based_cfpq(grammar, graph, {10}, {50}) == {(10, 50)}
    assert hellings_based_cfpq(grammar, graph) == {
        (10, 50),
        (20, 40),
        *((vertex, vertex) for vertex in graph),
    }


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiDiGraph])
@pytest.mark.parametrize(
    "text",
    [
        "S -> epsilon",
        "S -> A B\nA -> epsilon\nB -> epsilon",
        "S -> A\nA -> B\nB -> A | epsilon",
    ],
)
def test_epsilon_only_includes_isolated_vertices(graph_type, text):
    graph = graph_type()
    graph.add_nodes_from([-5, 40, 99])
    graph.add_edge(-5, 40, label="a")
    assert hellings_based_cfpq(CFG.from_text(text), graph) == {
        (-5, -5),
        (40, 40),
        (99, 99),
    }


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiDiGraph])
@pytest.mark.parametrize("text", ["S -> epsilon", "S -> a", "S -> S"])
def test_empty_graph(graph_type, text):
    assert hellings_based_cfpq(CFG.from_text(text), graph_type()) == set()


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiDiGraph])
def test_empty_language(graph_type):
    graph = graph_type()
    graph.add_edge(10, 20, label="a")
    assert hellings_based_cfpq(CFG.from_text("S -> A\nA -> S"), graph) == set()
    assert hellings_based_cfpq(CFG(start_symbol=Variable("S")), graph) == set()


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiDiGraph])
def test_recursive_ambiguous_grammar_on_cycle(graph_type):
    graph = graph_type()
    graph.add_edge(-5, 40, label="a")
    graph.add_edge(40, -5, label="a")
    grammar = CFG.from_text("S -> S S | a | epsilon")
    assert hellings_based_cfpq(grammar, graph) == {
        (-5, -5),
        (-5, 40),
        (40, -5),
        (40, 40),
    }


def test_parallel_edges_keep_every_label_and_deduplicate_pairs():
    graph = nx.MultiDiGraph()
    graph.add_edge(-5, 40, label="a")
    graph.add_edge(-5, 40, label="b")
    graph.add_edge(-5, 40, label="a")
    for query in ["S -> a", "S -> b", "S -> a | b"]:
        assert hellings_based_cfpq(CFG.from_text(query), graph) == {(-5, 40)}


def test_equal_textual_terminal_and_variable_values():
    start, variable = Variable("S"), Variable("A")
    grammar = CFG(
        start_symbol=start,
        productions=[
            Production(start, [Terminal("A"), variable, Terminal("b")]),
            Production(variable, [Terminal("c")]),
        ],
    )
    graph = nx.DiGraph()
    for source, target, label in [(10, 20, "A"), (20, 30, "c"), (30, 40, "b")]:
        graph.add_edge(source, target, label=label)
    assert hellings_based_cfpq(grammar, graph) == {(10, 40)}


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
    assert (
        hellings_based_cfpq(CFG.from_text("S -> a"), graph, starts, finals) == expected
    )


def test_epsilon_is_filtered_by_both_boundaries():
    graph = nx.DiGraph()
    graph.add_nodes_from([10, 20, 30, 99])
    assert hellings_based_cfpq(
        CFG.from_text("S -> epsilon"), graph, {10, 20, 99}, {20, 30, 99}
    ) == {(20, 20), (99, 99)}


@pytest.mark.parametrize("starts, finals", [({99}, None), (None, {99})])
def test_unknown_boundary_vertices(starts, finals):
    graph = nx.DiGraph()
    graph.add_node(10)
    with pytest.raises(ValueError, match="graph vertices"):
        hellings_based_cfpq(CFG.from_text("S -> epsilon"), graph, starts, finals)


def test_boundaries_do_not_prune_intermediate_vertices():
    graph = nx.DiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 30, label="b")
    assert hellings_based_cfpq(CFG.from_text("S -> a b"), graph, {10}, {30}) == {
        (10, 30)
    }


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
        assert hellings_based_cfpq(grammar, graph, starts, finals) == {
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
    "regex, text",
    [
        ("epsilon", "S -> epsilon"),
        ("a", "S -> a"),
        ("a b", "S -> a b"),
        ("a*", "S -> a S | epsilon"),
        ("(a | b)*", "S -> A S | epsilon\nA -> a | b"),
        ("a* b*", "S -> A B\nA -> a A | epsilon\nB -> b B | epsilon"),
        ("(a b)*", "S -> a b S | epsilon"),
    ],
)
def test_regular_grammars_match_existing_rpq(regex, text, seed):
    rng = random.Random(seed)
    vertices = [10, 30, 50]
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(vertices + [99])
    for _ in range(12):
        graph.add_edge(
            rng.choice(vertices), rng.choice(vertices), label=rng.choice("ab")
        )
    starts, finals = {10, 30, 99}, {30, 50, 99}
    expected = ms_bfs_based_rpq(regex, graph, starts, finals)
    assert hellings_based_cfpq(CFG.from_text(text), graph, starts, finals) == expected


def test_assignment_entry_points():
    from project import task6
    from project.weak_normal_form import cfg_to_weak_normal_form

    assert task6.hellings_based_cfpq is hellings_based_cfpq
    assert task6.cfg_to_weak_normal_form is cfg_to_weak_normal_form
    assert callable(task6.hellings_based_cfpq)
    assert callable(task6.cfg_to_weak_normal_form)
