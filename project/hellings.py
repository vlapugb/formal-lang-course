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

from project.weak_normal_form import cfg_to_weak_normal_form


def hellings_based_cfpq(
    cfg: CFG,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Return allowed endpoint pairs whose path label belongs to the CFG.

    The CFG may have arbitrary productions and a custom start symbol. The
    algorithm directly inspects DiGraph and MultiDiGraph edges with a ``label``.
    None or an empty boundary set selects every graph vertex; unknown boundary
    vertices raise ValueError. All vertices remain available as intermediate
    vertices. Empty paths are included when epsilon is in the language,
    including at isolated vertices. Inputs are not modified.
    """
    grammar = cfg_to_weak_normal_form(cfg)
    vertices = set(graph.nodes)
    starts = set(start_nodes) if start_nodes else vertices
    finals = set(final_nodes) if final_nodes else vertices
    if not (starts | finals) <= vertices:
        raise ValueError("Start and final states must be graph vertices")

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
    for source, target, data in graph.edges(data=True):
        label = data.get("label")
        for variable in terminal_rules.get(label, ()):
            add_fact(variable, source, target)

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
