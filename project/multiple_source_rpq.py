"""Regular path queries using sparse multiple-source breadth-first search."""

from networkx import MultiDiGraph
from scipy.sparse import csr_matrix

from project.automata_utils import graph_to_nfa, regex_to_dfa
from project.matrix_automata import AdjacencyMatrixFA


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Return vertex pairs whose connecting path matches ``regex``.

    For each query state, a Boolean CSR frontier has one row per source and
    one column per graph vertex. A BFS step multiplies these frontiers by
    the graph's label matrices along matching query transitions. Visited
    pairs are tracked separately for every source and query state.

    None or an empty boundary set selects all graph vertices, following the
    course convention. Zero-length paths are included if the query accepts
    the empty word. Unknown boundary vertices raise ValueError. Inputs are
    not modified. No product adjacency matrix or all-pairs closure is built.
    """
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    query_fa = AdjacencyMatrixFA(regex_to_dfa(regex))
    sources = sorted(graph_fa.start_states)
    if not sources or not graph_fa.final_states or not query_fa.start_states:
        return set()
    if not query_fa.final_states:
        return set()

    shape = (len(sources), graph_fa.num_states)
    frontier = [csr_matrix(shape, dtype=bool) for _ in query_fa.states]
    seed = csr_matrix(
        ([True] * len(sources), (range(len(sources)), sources)),
        shape=shape,
        dtype=bool,
    )
    for state in query_fa.start_states:
        frontier[state] = seed.copy()
    visited = [matrix.copy() for matrix in frontier]

    transitions = []
    for symbol in query_fa.matrices.keys() & graph_fa.matrices.keys():
        rows, columns = query_fa.matrices[symbol].nonzero()
        transitions.extend(
            (source, target, graph_fa.matrices[symbol])
            for source, target in zip(rows, columns)
        )

    while any(matrix.nnz for matrix in frontier):
        following = [csr_matrix(shape, dtype=bool) for _ in query_fa.states]
        for source, target, matrix in transitions:
            if frontier[source].nnz:
                following[target] = following[target] + frontier[source] @ matrix
        for state, candidates in enumerate(following):
            # The overlap is a subset of candidates, so != removes it without
            # a dense complement or unsupported Boolean sparse subtraction.
            following[state] = candidates != candidates.multiply(visited[state])
            following[state].eliminate_zeros()
            visited[state] = visited[state] + following[state]
        frontier = following

    reached = csr_matrix(shape, dtype=bool)
    for state in query_fa.final_states:
        reached = reached + visited[state]
    rows, columns = reached.nonzero()
    return {
        (graph_fa.states[sources[row]].value, graph_fa.states[column].value)
        for row, column in zip(rows, columns)
        if column in graph_fa.final_states
    }
