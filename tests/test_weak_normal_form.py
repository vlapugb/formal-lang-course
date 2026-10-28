"""Check weak normal form and preservation of grammar languages."""

from itertools import product

import pytest
from pyformlang.cfg import CFG, Production, Terminal, Variable

from project.weak_normal_form import cfg_to_weak_normal_form


def assert_weak_form(cfg):
    for production in cfg.productions:
        body = production.body
        assert isinstance(production.head, Variable)
        assert (
            not body
            or (len(body) == 1 and isinstance(body[0], Terminal))
            or (len(body) == 2 and all(isinstance(symbol, Variable) for symbol in body))
        )


def test_mixed_recursive_grammar_with_custom_start():
    grammar = CFG.from_text("Q -> a Q b c | epsilon", start_symbol=Variable("Q"))
    normalized = cfg_to_weak_normal_form(grammar)

    assert_weak_form(normalized)
    assert normalized.start_symbol == Variable("Q")
    for size in range(5):
        for word in product("abc", repeat=size):
            expected = not word or word == ("a", "b", "c")
            assert normalized.contains(word) == expected
            assert normalized.contains(word) == grammar.contains(word)
    assert normalized.contains(list("aabcbc"))
    assert not normalized.contains(list("aabbcc"))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("S -> epsilon", {()}),
        ("S -> A\nA -> B\nB -> A | epsilon", {()}),
        ("S -> A B\nA -> epsilon\nB -> b", {("b",)}),
        ("S -> A B\nA -> epsilon\nB -> epsilon", {()}),
        ("S -> A\nA -> S", set()),
        ("S -> A B\nA -> a", set()),
        ("S -> A\nA -> a b c", {("a", "b", "c")}),
    ],
)
def test_nullable_chains_and_empty_languages(text, expected):
    grammar = CFG.from_text(text)
    normalized = cfg_to_weak_normal_form(grammar)
    assert_weak_form(normalized)
    for size in range(4):
        for word in product("abc", repeat=size):
            assert normalized.contains(word) == (word in expected)
            assert normalized.contains(word) == grammar.contains(word)


def test_grammar_with_no_productions():
    grammar = CFG(start_symbol=Variable("Q"))
    normalized = cfg_to_weak_normal_form(grammar)
    assert_weak_form(normalized)
    assert normalized.start_symbol == Variable("Q")
    assert not normalized.contains([])
    assert not normalized.contains(["a"])


def test_start_symbol_on_right_and_non_start_epsilon():
    grammar = CFG.from_text("S -> A S | epsilon\nA -> a | epsilon")
    normalized = cfg_to_weak_normal_form(grammar)
    assert_weak_form(normalized)
    assert normalized.start_symbol == grammar.start_symbol
    for size in range(6):
        assert normalized.contains(["a"] * size)
    assert not normalized.contains(["b"])


def acyclic_words(grammar, variable):
    """Enumerate finite derivations with explicit terminal/variable kinds."""
    words = set()
    for production in grammar.productions:
        if production.head != variable:
            continue
        choices = [
            {(symbol.value,)}
            if isinstance(symbol, Terminal)
            else acyclic_words(grammar, symbol)
            for symbol in production.body
        ]
        words.update(sum(pieces, ()) for pieces in product(*choices))
    return words


def test_helper_names_do_not_change_original_variables_or_terminals():
    start = Variable("S")
    variables = [Variable(name) for name in ("__wcnf_0", "__wcnf_1", "C#CNF#1")]
    grammar = CFG(
        start_symbol=start,
        productions=[
            Production(start, [Terminal("a"), *variables, Terminal("__wcnf_2")]),
            *(Production(variable, [Terminal("b")]) for variable in variables),
        ],
    )
    normalized = cfg_to_weak_normal_form(grammar)
    assert_weak_form(normalized)
    assert acyclic_words(normalized, start) == {("a", "b", "b", "b", "__wcnf_2")}


def test_terminal_and_variable_with_equal_text_remain_distinct():
    start, variable = Variable("S"), Variable("A")
    grammar = CFG(
        start_symbol=start,
        productions=[
            Production(start, [Terminal("A"), variable, Terminal("b")]),
            Production(variable, [Terminal("c")]),
        ],
    )
    normalized = cfg_to_weak_normal_form(grammar)
    assert_weak_form(normalized)
    # pyformlang's CNF membership oracle is unreliable for equal textual values.
    assert acyclic_words(normalized, start) == {("A", "c", "b")}


def test_output_is_independent_and_input_is_unchanged():
    grammar = CFG.from_text("S -> a A b | epsilon\nA -> c")
    original = (
        set(grammar.productions),
        set(grammar.variables),
        set(grammar.terminals),
        grammar.start_symbol,
    )
    bodies = [(production, list(production.body)) for production in grammar.productions]
    normalized = cfg_to_weak_normal_form(grammar)
    assert normalized is not grammar
    assert (
        set(grammar.productions),
        set(grammar.variables),
        set(grammar.terminals),
        grammar.start_symbol,
    ) == original
    for production in normalized.productions:
        production.body.clear()
    normalized.variables.clear()
    normalized.terminals.clear()
    assert all(production.body == body for production, body in bodies)
    assert grammar.variables == original[1]
    assert grammar.terminals == original[2]
