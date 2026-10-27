"""Algoritmo de grafo BPE, resolução de ancestrais transitivos e fechamento sob merges."""


def build_dependency_graph(
    merges: list[tuple[str, str]],
) -> dict[str, tuple[str, str]]:
    """Constrói o grafo direcionado de dependências a partir das regras de merge BPE.

    No algoritmo BPE, cada merge unifica dois tokens existentes A e B para formar
    o token composto C = A + B. O grafo mapeia o token resultante para a tupla de
    seus dois constituintes imediatos:
        C -> (A, B)

    Args:
        merges: Lista de tuplas (A, B) representando regras de merge em ordem de aplicação.

    Returns:
        Dicionário mapeando cada token composto resultante (A + B) para a tupla (A, B).
    """
    graph: dict[str, tuple[str, str]] = {}
    for left, right in merges:
        graph[left + right] = (left, right)
    return graph


def compute_ancestors(
    token: str,
    dep_graph: dict[str, tuple[str, str]],
    memo: dict[str, set[str]],
) -> set[str]:
    """Calcula recursivamente o fechamento transitivo de ancestrais de um token no grafo BPE.

    Ancestrais(C) = {A, B} U Ancestrais(A) U Ancestrais(B).
    Se o token for elementar (não estiver em dep_graph), o conjunto de ancestrais é vazio.
    Utiliza memoização in-place no dicionário `memo` para complexidade O(1) amortizada
    em chamadas subsequentes sobre os ~248K tokens do vocabulário original.

    Args:
        token: Token cujo conjunto de ancestrais transitivos será calculado.
        dep_graph: Grafo de dependências mapeando token composto para (pai_esq, pai_dir).
        memo: Dicionário de memoização (token -> conjunto de ancestrais transitivos).

    Returns:
        Conjunto contendo todos os ancestrais transitivos do token.
    """
    if token in memo:
        return memo[token]

    if token not in dep_graph:
        ancestors: set[str] = set()
        memo[token] = ancestors
        return ancestors

    left, right = dep_graph[token]
    ancestors = {left, right}
    ancestors.update(compute_ancestors(left, dep_graph, memo))
    ancestors.update(compute_ancestors(right, dep_graph, memo))
    memo[token] = ancestors
    return ancestors


def is_merge_closed(
    vocab: set[str],
    merges: list[tuple[str, str]],
) -> bool:
    """Verifica se um determinado vocabulário é estritamente fechado sob uma lista de merges BPE.

    Um vocabulário é fechado sob merges se, para cada regra (A, B) cujo resultado
    C = A + B pertença ao vocabulário, ambos os predecessores A e B obrigatoriamente
    também pertençam ao vocabulário.

    Args:
        vocab: Conjunto de tokens que compõem o vocabulário (ex: Vocab_96k).
        merges: Lista de tuplas de merge (A, B).

    Returns:
        True se todas as dependências imediatas estiverem no vocabulário, False caso contrário.
    """
    for left, right in merges:
        merged = left + right
        if merged in vocab:
            if left not in vocab or right not in vocab:
                return False
    return True
