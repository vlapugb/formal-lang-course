"""Language-preserving conversion to weak Chomsky normal form."""

from pyformlang.cfg import CFG, Production, Terminal, Variable


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    """Return an independent CFG with the same language and start symbol.

    Rules have the forms A -> B C, A -> a, or A -> epsilon. Empty words and
    nullable constituents are retained, including epsilon inherited through
    chain rules. The supplied grammar's productions and symbol sets are
    unchanged; helper variables are fresh with respect to the input symbols.
    """
    occupied = {symbol.value for symbol in list(cfg.variables) + list(cfg.terminals)}
    next_index = 0

    def fresh_variable() -> Variable:
        nonlocal next_index
        while True:
            name = f"__wcnf_{next_index}"
            next_index += 1
            if name not in occupied:
                occupied.add(name)
                return Variable(name)

    terminal_variables = {}
    productions = []
    for production in cfg.productions:
        body = list(production.body)
        # Replace only terminals in compound bodies.
        if len(body) > 1:
            for index, symbol in enumerate(body):
                if isinstance(symbol, Terminal):
                    if symbol not in terminal_variables:
                        variable = fresh_variable()
                        terminal_variables[symbol] = variable
                        productions.append(Production(variable, [symbol]))
                    body[index] = terminal_variables[symbol]

        head = production.head
        # Expand long bodies into the rightward chain of binary productions.
        for symbol in body[:-2]:
            following = fresh_variable()
            productions.append(Production(head, [symbol, following]))
            head = following
        productions.append(Production(head, body[-2:]))

    # The public method implements reflexive chain-pair closure and copies
    # every non-chain production, including empty bodies.
    working = (
        CFG(
            variables=set(cfg.variables),
            terminals=set(cfg.terminals),
            start_symbol=cfg.start_symbol,
            productions=productions,
        )
        .eliminate_unit_productions()
        .remove_useless_symbols()
    )
    return CFG(
        variables=set(working.variables),
        terminals=set(working.terminals),
        start_symbol=cfg.start_symbol,
        productions={
            Production(production.head, list(production.body))
            for production in working.productions
        },
    )
