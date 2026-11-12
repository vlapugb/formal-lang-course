"""Check descriptor-based CFPQ over recursive automata."""

import random
from copy import deepcopy

import networkx as nx
import pytest
from pyformlang.cfg import CFG, Variable
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol
from pyformlang.rsa import Box, RecursiveAutomaton

from project.gll_cfpq import gll_based_cfpq
from project.hellings import hellings_based_cfpq
from project.tensor_cfpq import cfg_to_rsm, ebnf_to_rsm, tensor_based_cfpq


def _path(*labels):
    graph = nx.MultiDiGraph()
    graph.add_node(0)
    for index, label in enumerate(labels):
        graph.add_edge(index, index + 1, label=label)
    return graph


def _box(label, symbol):
    automaton = NondeterministicFiniteAutomaton()
    automaton.add_start_state(0)
    automaton.add_final_state(1)
    automaton.add_transition(0, symbol, 1)
    return Box(automaton, Symbol(label))


def test_epsilon_queries_include_isolated_vertices():
    graph = nx.DiGraph()
    graph.add_nodes_from([10, 99])

    direct = gll_based_cfpq(cfg_to_rsm(CFG.from_text("S -> epsilon")), graph)
    indirect = gll_based_cfpq(cfg_to_rsm(CFG.from_text("S -> A\nA -> epsilon")), graph)

    assert direct == {(10, 10), (99, 99)}
    assert indirect == direct


def test_boxes_with_overlapping_state_names_remain_distinct():
    rsm = RecursiveAutomaton(
        labels={Symbol("S"), Symbol("A")},
        initial_label=Symbol("S"),
        boxes={_box("S", "A"), _box("A", "a")},
    )
    graph = nx.DiGraph()
    graph.add_edge(10, 20, label="a")

    assert gll_based_cfpq(rsm, graph) == {(10, 20)}


def test_nested_recursive_language():
    rsm = cfg_to_rsm(CFG.from_text("S -> a S b | a b"))

    assert gll_based_cfpq(rsm, _path("a", "a", "b", "b")) == {(0, 4), (1, 3)}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("S -> S a | a", {(0, 1), (1, 2), (0, 2)}),
        ("S -> S S | a", {(0, 1), (1, 2), (0, 2)}),
        ("S -> S | a", {(0, 1), (1, 2)}),
    ],
)
def test_left_recursion_terminates(text, expected):
    rsm = cfg_to_rsm(CFG.from_text(text))

    assert gll_based_cfpq(rsm, _path("a", "a")) == expected


def test_cyclic_graph_with_nested_calls():
    graph = nx.DiGraph()
    graph.add_edge(0, 0, label="a")
    graph.add_edge(0, 1, label="b")
    graph.add_edge(1, 1, label="b")
    rsm = cfg_to_rsm(CFG.from_text("S -> a S b | a b"))

    assert gll_based_cfpq(rsm, graph) == {(0, 1)}


def test_call_added_after_callee_returned_resumes():
    # Both callers share the A call at vertex 0, so whichever caller is
    # registered after the call has returned must still be resumed.
    rsm = cfg_to_rsm(CFG.from_text("S -> A c | B\nB -> A d\nA -> a"))

    assert gll_based_cfpq(rsm, _path("a", "c"), {0}) == {(0, 2)}
    assert gll_based_cfpq(rsm, _path("a", "d"), {0}) == {(0, 2)}


def test_mutually_recursive_language():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="b")
    graph.add_edge(1, 2, label="a")
    rsm = cfg_to_rsm(CFG.from_text("S -> A\nA -> B a | epsilon\nB -> A b | b"))

    assert gll_based_cfpq(rsm, graph) == {(0, 0), (1, 1), (2, 2), (0, 2)}


def test_extended_grammar_with_star():
    rsm = ebnf_to_rsm("S -> a* B\nB -> b")

    assert gll_based_cfpq(rsm, _path("a", "a", "b")) == {(0, 3), (1, 3), (2, 3)}


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
    rsm = cfg_to_rsm(CFG.from_text("S -> a"))

    assert gll_based_cfpq(rsm, graph, starts, finals) == expected


@pytest.mark.parametrize("starts, finals", [({99}, None), (None, {99})])
def test_unknown_boundary_vertices(starts, finals):
    graph = nx.DiGraph()
    graph.add_node(10)
    rsm = cfg_to_rsm(CFG.from_text("S -> epsilon"))

    with pytest.raises(ValueError, match="graph vertices"):
        gll_based_cfpq(rsm, graph, starts, finals)


def test_boundaries_do_not_prune_intermediate_vertices():
    graph = nx.DiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 30, label="b")
    rsm = cfg_to_rsm(CFG.from_text("S -> a b"))

    assert gll_based_cfpq(rsm, graph, {10}, {30}) == {(10, 30)}


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiDiGraph])
def test_empty_graph_and_empty_language(graph_type):
    assert (
        gll_based_cfpq(cfg_to_rsm(CFG.from_text("S -> epsilon")), graph_type()) == set()
    )

    graph = graph_type()
    graph.add_edge(10, 20, label="a")
    assert gll_based_cfpq(cfg_to_rsm(CFG(start_symbol=Variable("Q"))), graph) == set()
    assert gll_based_cfpq(cfg_to_rsm(CFG()), graph) == set()


def test_parallel_edges_and_noncontiguous_identifiers():
    graph = nx.MultiDiGraph()
    graph.add_edge(-5, 40, label="a")
    graph.add_edge(-5, 40, label="b")
    graph.add_edge(-5, 40, label="a")
    rsm = cfg_to_rsm(CFG.from_text("S -> a | b"))

    assert gll_based_cfpq(rsm, graph) == {(-5, 40)}


def test_repeated_queries_preserve_all_inputs():
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, label="a", metadata={"weight": 2})
    graph.add_edge(20, 30, label="b")
    graph.add_node(99, description="isolated")
    original_graph = deepcopy(graph)
    rsm = cfg_to_rsm(CFG.from_text("S -> a A | epsilon\nA -> b"))
    original_rsm = deepcopy(rsm)
    starts, finals = {10, 99}, {30, 99}

    for _ in range(2):
        assert gll_based_cfpq(rsm, graph, starts, finals) == {(10, 30), (99, 99)}
        assert nx.utils.graphs_equal(graph, original_graph)
        assert rsm == original_rsm
        assert rsm.initial_label == original_rsm.initial_label
        assert starts == {10, 99}
        assert finals == {30, 99}


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    "text",
    [
        "S -> a",
        "S -> a S b | epsilon",
        "S -> S S | a | epsilon",
        "S -> S a | b",
        "S -> A B\nA -> a A | epsilon\nB -> b B | epsilon",
    ],
)
def test_matches_hellings_and_tensor_on_random_graphs(text, seed):
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

    gll = gll_based_cfpq(cfg_to_rsm(grammar), graph, starts, finals)

    assert gll == hellings_based_cfpq(grammar, graph, starts, finals)
    assert gll == tensor_based_cfpq(cfg_to_rsm(grammar), graph, starts, finals)
