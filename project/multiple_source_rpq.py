"""Regular path queries using sparse multiple-source breadth-first search."""

from networkx import MultiDiGraph
from scipy.sparse import csr_matrix, eye, kron

from project.automata_utils import graph_to_nfa, regex_to_dfa
from project.matrix_automata import AdjacencyMatrixFA


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Return vertex pairs whose connecting path matches ``regex``.

    The Boolean CSR frontier has one row per (source, query state) pair and
    one column per graph vertex. For each label, a BFS step computes
    Q.T @ frontier @ G, with one query matrix Q block per source and the
    graph matrix G. Visited entries are removed before the next step.

    None or an empty boundary set selects all graph vertices, following the
    course convention. Zero-length paths are included if the query accepts
    the empty word. Unknown boundary vertices raise ValueError. Inputs are
    not modified. No graph/query product adjacency matrix or all-pairs
    closure is built.
    """
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    query_fa = AdjacencyMatrixFA(regex_to_dfa(regex))
    sources = sorted(graph_fa.start_states)
    if not sources or not graph_fa.final_states or not query_fa.start_states:
        return set()
    if not query_fa.final_states:
        return set()

    query_size = query_fa.num_states
    shape = (len(sources) * query_size, graph_fa.num_states)
    seed_rows, seed_columns = [], []
    for source_index, vertex in enumerate(sources):
        for state in query_fa.start_states:
            seed_rows.append(source_index * query_size + state)
            seed_columns.append(vertex)
    frontier = csr_matrix(
        ([True] * len(seed_rows), (seed_rows, seed_columns)),
        shape=shape,
        dtype=bool,
    )
    visited = frontier.copy()

    source_identity = eye(len(sources), format="csr", dtype=bool)
    transitions = [
        (
            kron(source_identity, query_fa.matrices[symbol].T, format="csr"),
            graph_fa.matrices[symbol],
        )
        for symbol in query_fa.matrices.keys() & graph_fa.matrices.keys()
    ]

    while frontier.nnz:
        following = csr_matrix(shape, dtype=bool)
        for query_transpose, graph_matrix in transitions:
            following = following + query_transpose @ frontier @ graph_matrix
        following = following != following.multiply(visited)
        following.eliminate_zeros()
        visited = visited + following
        frontier = following

    rows, columns = visited.nonzero()
    return {
        (
            graph_fa.states[sources[row // query_size]].value,
            graph_fa.states[column].value,
        )
        for row, column in zip(rows, columns)
        if row % query_size in query_fa.final_states and column in graph_fa.final_states
    }
