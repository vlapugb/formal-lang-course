"""Check recursive-automaton conversion and tensor-based CFPQ."""

import random
from copy import deepcopy

import networkx as nx
import pytest
from pyformlang.cfg import CFG, Variable
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol
from pyformlang.rsa import Box, RecursiveAutomaton

from project.hellings import hellings_based_cfpq
from project.tensor_cfpq import cfg_to_rsm, ebnf_to_rsm, tensor_based_cfpq


def test_cfg_conversion_preserves_start_and_groups_productions():
    cfg = CFG.from_text("Q -> a Q b | epsilon\nA -> a", start_symbol=Variable("Q"))
    rsm = cfg_to_rsm(cfg)

    assert rsm.initial_label == Symbol("Q")
    assert rsm.labels == {Symbol("Q"), Symbol("A")}
    assert rsm.get_number_of_boxes() == 2
    box = rsm.get_box(Symbol("Q")).dfa
    assert box.accepts([])
    assert box.accepts(["a", "Q", "b"])


def test_ebnf_conversion_supports_regular_operators():
    rsm = ebnf_to_rsm("S -> (a | b)* c | $")
    box = rsm.get_box(Symbol("S")).dfa

    assert rsm.initial_label == Symbol("S")
    assert box.accepts([])
    assert box.accepts(["a", "b", "a", "c"])
    assert not box.accepts(["a", "b"])


def test_empty_cfg_conversion_produces_empty_language():
    rsm = cfg_to_rsm(CFG(start_symbol=Variable("Q")))

    assert rsm.initial_label == Symbol("Q")
    box = rsm.get_box(Symbol("Q")).dfa
    assert not box.accepts([])
    assert not box.accepts(["a"])
    assert cfg_to_rsm(CFG()).boxes == {}


def test_epsilon_queries_include_isolated_vertices():
    graph = nx.DiGraph()
    graph.add_nodes_from([10, 99])

    direct = tensor_based_cfpq(cfg_to_rsm(CFG.from_text("S -> epsilon")), graph)
    indirect = tensor_based_cfpq(
        cfg_to_rsm(CFG.from_text("S -> A\nA -> epsilon")), graph
    )

    assert direct == {(10, 10), (99, 99)}
    assert indirect == direct


def _box(label, symbol):
    automaton = NondeterministicFiniteAutomaton()
    automaton.add_start_state(0)
    automaton.add_final_state(1)
    automaton.add_transition(0, symbol, 1)
    return Box(automaton, Symbol(label))


def test_boxes_with_overlapping_state_names_remain_distinct():
    rsm = RecursiveAutomaton(
        labels={Symbol("S"), Symbol("A")},
        initial_label=Symbol("S"),
        boxes={_box("S", "A"), _box("A", "a")},
    )
    graph = nx.DiGraph()
    graph.add_edge(10, 20, label="a")

    assert tensor_based_cfpq(rsm, graph) == {(10, 20)}


def test_nested_recursive_language():
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            (0, 1, {"label": "a"}),
            (1, 2, {"label": "a"}),
            (2, 3, {"label": "b"}),
            (3, 4, {"label": "b"}),
        ]
    )
    rsm = cfg_to_rsm(CFG.from_text("S -> a S b | a b"))

    assert tensor_based_cfpq(rsm, graph) == {(0, 4), (1, 3)}


def test_mutually_recursive_language():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="b")
    graph.add_edge(1, 2, label="a")
    rsm = cfg_to_rsm(CFG.from_text("S -> A\nA -> B a | epsilon\nB -> A b | b"))

    assert tensor_based_cfpq(rsm, graph) == {
        (0, 0),
        (1, 1),
        (2, 2),
        (0, 2),
    }


def test_ambiguous_grammar_returns_each_pair_once():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="a")
    rsm = cfg_to_rsm(CFG.from_text("S -> S S | a"))

    assert tensor_based_cfpq(rsm, graph) == {(0, 1), (1, 2), (0, 2)}


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

    assert tensor_based_cfpq(rsm, graph, starts, finals) == expected


@pytest.mark.parametrize("starts, finals", [({99}, None), (None, {99})])
def test_unknown_boundary_vertices(starts, finals):
    graph = nx.DiGraph()
    graph.add_node(10)
    rsm = cfg_to_rsm(CFG.from_text("S -> epsilon"))

    with pytest.raises(ValueError, match="graph vertices"):
        tensor_based_cfpq(rsm, graph, starts, finals)


def test_boundaries_do_not_prune_intermediate_vertices():
    graph = nx.DiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 30, label="b")
    rsm = cfg_to_rsm(CFG.from_text("S -> a b"))

    assert tensor_based_cfpq(rsm, graph, {10}, {30}) == {(10, 30)}


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiDiGraph])
def test_empty_graph_and_empty_language(graph_type):
    assert (
        tensor_based_cfpq(cfg_to_rsm(CFG.from_text("S -> epsilon")), graph_type())
        == set()
    )

    graph = graph_type()
    graph.add_edge(10, 20, label="a")
    empty = cfg_to_rsm(CFG(start_symbol=Variable("Q")))
    assert tensor_based_cfpq(empty, graph) == set()
    assert tensor_based_cfpq(cfg_to_rsm(CFG()), graph) == set()


def test_parallel_edges_and_noncontiguous_identifiers():
    graph = nx.MultiDiGraph()
    graph.add_edge(-5, 40, label="a")
    graph.add_edge(-5, 40, label="b")
    graph.add_edge(-5, 40, label="a")
    rsm = cfg_to_rsm(CFG.from_text("S -> a | b"))

    assert tensor_based_cfpq(rsm, graph) == {(-5, 40)}


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
        assert tensor_based_cfpq(rsm, graph, starts, finals) == {
            (10, 30),
            (99, 99),
        }
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

    assert tensor_based_cfpq(
        cfg_to_rsm(deepcopy(grammar)), deepcopy(graph), starts, finals
    ) == hellings_based_cfpq(deepcopy(grammar), deepcopy(graph), starts, finals)
