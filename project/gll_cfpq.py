"""Context-free path queries over recursive automata using descriptor search.

A descriptor ``(state, vertex, node)`` says that the RSM state ``state`` was
reached at graph vertex ``vertex`` inside the call represented by stack node
``node``. A stack node ``(label, vertex)`` is a call of box ``label`` that
started at ``vertex``; it is shared by every caller, and its edges store the
return states with their caller nodes. Ending a call at a vertex records that
vertex as popped for the node, so callers added later resume from it too.
"""

from collections import defaultdict, deque
from collections.abc import Hashable, Iterable

from networkx import DiGraph
from pyformlang.finite_automaton import Symbol
from pyformlang.rsa import RecursiveAutomaton

from project.automata_utils import graph_to_nfa

RsmState = tuple[Symbol, Hashable]
StackNode = tuple[Symbol, Hashable]
Descriptor = tuple[RsmState, Hashable, StackNode]


def targets(value) -> Iterable:
    """Return transition targets whether stored as one state or a set."""
    return value if isinstance(value, set) else (value,)


class RsmIndex:
    """Box boundaries and outgoing transitions with box-qualified states."""

    def __init__(self, rsm: RecursiveAutomaton):
        self.starts: dict[Symbol, set[RsmState]] = {}
        self.finals: set[RsmState] = set()
        self.terminals: dict[RsmState, list[tuple[Symbol, RsmState]]] = defaultdict(
            list
        )
        self.calls: dict[RsmState, list[tuple[Symbol, RsmState]]] = defaultdict(list)

        for label, box in rsm.boxes.items():
            self.starts[label] = {(label, state) for state in box.start_state}
            self.finals.update((label, state) for state in box.final_states)
            for source, transitions in box.dfa.to_dict().items():
                for symbol, value in transitions.items():
                    kind = self.calls if symbol in rsm.boxes else self.terminals
                    for target in targets(value):
                        kind[(label, source)].append((symbol, (label, target)))


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Return endpoint pairs whose connecting paths are accepted by ``rsm``.

    ``None`` or an empty start/final set selects all graph vertices; unknown
    boundary vertices raise ``ValueError``. All vertices may appear inside
    paths. Empty paths are included when the initial label derives epsilon.
    The input automaton, graph, and boundary sets are not modified.
    """
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    starts = {state.value for state in graph_nfa.start_states}
    finals = {state.value for state in graph_nfa.final_states}
    edges: dict[Hashable, dict[Symbol, set]] = defaultdict(lambda: defaultdict(set))
    for source, transitions in graph_nfa.to_dict().items():
        for symbol, value in transitions.items():
            edges[source.value][symbol].update(
                target.value for target in targets(value)
            )

    index = RsmIndex(rsm)
    initial = rsm.initial_label
    if initial not in index.starts:
        return set()

    returns: dict[StackNode, set[tuple[RsmState, StackNode]]] = defaultdict(set)
    popped: dict[StackNode, set[Hashable]] = defaultdict(set)
    visited: set[Descriptor] = set()
    pending: deque[Descriptor] = deque()

    def add(state: RsmState, vertex: Hashable, node: StackNode) -> None:
        descriptor = (state, vertex, node)
        if descriptor not in visited:
            visited.add(descriptor)
            pending.append(descriptor)

    def call(label: Symbol, vertex: Hashable) -> StackNode:
        node = (label, vertex)
        if node not in returns:
            returns[node] = set()
            for state in index.starts[label]:
                add(state, vertex, node)
        return node

    for vertex in starts:
        call(initial, vertex)

    while pending:
        state, vertex, node = pending.popleft()

        if state in index.finals and vertex not in popped[node]:
            popped[node].add(vertex)
            for return_state, caller in returns[node]:
                add(return_state, vertex, caller)

        for symbol, target in index.terminals.get(state, ()):
            for successor in edges[vertex].get(symbol, ()):
                add(target, successor, node)

        for label, return_state in index.calls.get(state, ()):
            callee = call(label, vertex)
            frame = (return_state, node)
            if frame not in returns[callee]:
                returns[callee].add(frame)
                for end in popped[callee]:
                    add(return_state, end, node)

    return {
        (source, target)
        for source in starts
        for target in popped[(initial, source)]
        if target in finals
    }
