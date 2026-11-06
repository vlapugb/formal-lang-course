"""Construct recursive automata and evaluate context-free path queries."""

from collections import defaultdict

from networkx import DiGraph
from pyformlang.cfg import CFG
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from pyformlang.regular_expression import Regex
from pyformlang.rsa import RecursiveAutomaton
from scipy.sparse import csr_matrix, eye

from project.automata_utils import graph_to_nfa
from project.matrix_automata import AdjacencyMatrixFA, intersect_automata


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    """Build a recursive automaton representing ``cfg``."""
    if not cfg.productions and cfg.start_symbol is not None:
        return RecursiveAutomaton.from_regex(Regex(""), cfg.start_symbol)
    return RecursiveAutomaton.from_text(cfg.to_text(), cfg.start_symbol)


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    """Build a recursive automaton from productions in EBNF syntax."""
    return RecursiveAutomaton.from_text(ebnf)


def _rsm_to_fa(
    rsm: RecursiveAutomaton,
) -> tuple[AdjacencyMatrixFA, dict[tuple[int, int], set[Symbol]]]:
    """Combine boxes into matrices and retain their accepting boundaries."""
    combined = NondeterministicFiniteAutomaton(
        states={
            State((label, state))
            for label, box in rsm.boxes.items()
            for state in box.dfa.states
        }
    )
    for label, box in rsm.boxes.items():
        for source, transitions in box.dfa.to_dict().items():
            for symbol, target in transitions.items():
                combined.add_transition(
                    State((label, source)), symbol, State((label, target))
                )

    automaton = AdjacencyMatrixFA(combined)
    boundaries = defaultdict(set)
    for label, box in rsm.boxes.items():
        starts = {
            automaton.state_indices[State((label, state))] for state in box.start_state
        }
        finals = {
            automaton.state_indices[State((label, state))] for state in box.final_states
        }
        for start in starts:
            for final in finals:
                boundaries[(start, final)].add(label)
    return automaton, dict(boundaries)


def _derived_edges(
    closure: csr_matrix,
    boundaries: dict[tuple[int, int], set[Symbol]],
    graph_size: int,
) -> dict[Symbol, csr_matrix]:
    """Collect graph edges derived from accepting paths in the product."""
    coordinates = defaultdict(list)
    rows, columns = closure.nonzero()
    for row, column in zip(rows, columns):
        labels = boundaries.get((row // graph_size, column // graph_size))
        if not labels:
            continue
        for label in labels:
            coordinates[label].append((row % graph_size, column % graph_size))

    edges = {}
    for label, pairs in coordinates.items():
        rows, columns = zip(*pairs)
        edges[label] = csr_matrix(
            ([True] * len(rows), (rows, columns)),
            shape=(graph_size, graph_size),
            dtype=bool,
        )
    return edges


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Return endpoint pairs whose connecting paths are accepted by ``rsm``.

    ``None`` or an empty start/final set selects all graph vertices; unknown
    boundary vertices raise ``ValueError``. Empty paths are included when the
    initial label derives epsilon. The input automaton, graph, and boundary
    sets are not modified.
    """
    rsm_fa, boundaries = _rsm_to_fa(rsm)
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    graph_size = graph_fa.num_states

    for label, box in rsm.boxes.items():
        if box.start_state & box.final_states:
            current = graph_fa.matrices.get(
                label, csr_matrix((graph_size, graph_size), dtype=bool)
            )
            graph_fa.matrices[label] = current.maximum(
                eye(graph_size, format="csr", dtype=bool)
            )

    while True:
        product = intersect_automata(rsm_fa, graph_fa)
        derived = _derived_edges(product.transitive_closure(), boundaries, graph_size)
        changed = False
        for label, candidate in derived.items():
            current = graph_fa.matrices.get(
                label, csr_matrix((graph_size, graph_size), dtype=bool)
            )
            updated = current.maximum(candidate)
            if updated.nnz != current.nnz:
                graph_fa.matrices[label] = updated
                changed = True
        if not changed:
            break

    start_matrix = graph_fa.matrices.get(rsm.initial_label)
    if start_matrix is None:
        return set()
    rows, columns = start_matrix.nonzero()
    return {
        (graph_fa.states[row].value, graph_fa.states[column].value)
        for row, column in zip(rows, columns)
        if row in graph_fa.start_states and column in graph_fa.final_states
    }
