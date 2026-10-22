"""Context-free path queries using the Hellings algorithm.

A fact (A, u, v) means that a path from u to v spells a word derived from A.
Initialization covers terminal edges and epsilon at every vertex. Processing
each fact composes it with both incoming and outgoing facts using binary rules.
Duplicate suppression bounds the number of pending insertions by |N'| * |V|^2,
ensuring termination on finite inputs.
"""

from collections import defaultdict, deque

from networkx import DiGraph
from pyformlang.cfg import CFG

from project.automata_utils import graph_to_nfa
from project.weak_normal_form import cfg_to_weak_normal_form


def hellings_based_cfpq(
    cfg: CFG,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Return allowed endpoint pairs whose path label belongs to the CFG.

    The CFG may have arbitrary productions and a custom start symbol. The
    existing graph adapter supports DiGraph and MultiDiGraph with a ``label``
    on each edge. None or an empty boundary set selects every graph vertex;
    unknown boundary vertices raise ValueError. All vertices remain available
    as intermediate vertices. Empty paths are included when epsilon is in the
    language, including at isolated vertices. Inputs are not modified.
    """
    grammar = cfg_to_weak_normal_form(cfg)
    automaton = graph_to_nfa(graph, start_nodes, final_nodes)
    vertices = {state.value for state in automaton.states}
    starts = {state.value for state in automaton.start_states}
    finals = {state.value for state in automaton.final_states}

    terminal_rules = defaultdict(set)
    binary_rules = defaultdict(set)
    empty_rules = set()
    for production in grammar.productions:
        body = production.body
        if not body:
            empty_rules.add(production.head)
        elif len(body) == 1:
            terminal_rules[body[0].value].add(production.head)
        else:
            binary_rules[tuple(body)].add(production.head)

    discovered = set()
    pending = deque()
    incoming = defaultdict(set)
    outgoing = defaultdict(set)

    def add_fact(variable, source, target):
        fact = (variable, source, target)
        if fact not in discovered:
            discovered.add(fact)
            pending.append(fact)
            incoming[target].add((variable, source))
            outgoing[source].add((variable, target))

    # Initialize epsilon facts and matching terminal edges.
    for variable in empty_rules:
        for vertex in vertices:
            add_fact(variable, vertex, vertex)
    for source, transitions in automaton.to_dict().items():
        for symbol, targets in transitions.items():
            for variable in terminal_rules.get(symbol.value, ()):
                for target in targets:
                    add_fact(variable, source.value, target.value)

    while pending:
        variable, source, target = pending.popleft()
        # Incoming facts: A -> C B, with the pending fact labeled B.
        for left, predecessor in tuple(incoming[source]):
            for head in binary_rules.get((left, variable), ()):
                add_fact(head, predecessor, target)
        # Outgoing facts: A -> B C. Snapshots permit insertions on self-loops.
        for right, successor in tuple(outgoing[target]):
            for head in binary_rules.get((variable, right), ()):
                add_fact(head, source, successor)

    return {
        (source, target)
        for variable, source, target in discovered
        if variable == grammar.start_symbol and source in starts and target in finals
    }
