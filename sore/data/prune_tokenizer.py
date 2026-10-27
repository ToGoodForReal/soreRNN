"""Algoritmo de grafo BPE, resolução de ancestrais transitivos e seleção gulosa com fechamento."""

import re

_HEX_BYTE_RE = re.compile(r"^<0x([0-9a-fA-F]{2})>$")


def _piece_to_str(piece: str) -> str:
    """Converte representação hexadecimal de byte (<0xHH>) para caractere, ou retorna a peça original."""
    m = _HEX_BYTE_RE.match(piece)
    if m:
        return chr(int(m.group(1), 16))
    return piece


def _merge_result(left: str, right: str) -> str:
    """Determina o token resultante da concatenação de duas peças BPE, considerando fallback de bytes."""
    m_left = _HEX_BYTE_RE.match(left)
    m_right = _HEX_BYTE_RE.match(right)
    if m_left or m_right:
        s_left = _piece_to_str(left)
        s_right = _piece_to_str(right)
        return s_left + s_right
    return left + right


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
        Dicionário mapeando cada token composto resultante para a tupla (A, B).
    """
    graph: dict[str, tuple[str, str]] = {}
    for left, right in merges:
        graph[left + right] = (left, right)
        res = _merge_result(left, right)
        if res != left + right:
            graph[res] = (left, right)
    return graph


def compute_ancestors(
    token: str,
    dep_graph: dict[str, tuple[str, str]],
    memo: dict[str, set[str]] | None = None,
) -> set[str]:
    """Calcula recursivamente o fechamento transitivo de ancestrais de um token no grafo BPE.

    Ancestrais(C) = {A, B} U Ancestrais(A) U Ancestrais(B).
    Se o token for elementar (não estiver em dep_graph), o conjunto de ancestrais é vazio.
    Utiliza memoização in-place no dicionário `memo` para complexidade O(1) amortizada
    em chamadas subsequentes sobre os ~248K tokens do vocabulário original.

    Args:
        token: Token cujo conjunto de ancestrais transitivos será calculado.
        dep_graph: Grafo de dependências mapeando token composto para (pai_esq, pai_dir).
        memo: Dicionário opcional de memoização (token -> conjunto de ancestrais transitivos).

    Returns:
        Conjunto contendo todos os ancestrais transitivos do token.
    """
    if memo is None:
        memo = {}

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
    C pertença ao vocabulário, ambos os predecessores A e B obrigatoriamente
    também pertençam ao vocabulário.

    Args:
        vocab: Conjunto de tokens que compõem o vocabulário (ex: Vocab_96k).
        merges: Lista de tuplas de merge (A, B).

    Returns:
        True se todas as dependências imediatas estiverem no vocabulário, False caso contrário.
    """
    for left, right in merges:
        candidates = {left + right, _merge_result(left, right)}
        for merged in candidates:
            if merged in vocab:
                if left not in vocab or right not in vocab:
                    return False
    return True


def select_pruned_vocab(
    frequencies: dict[str, int],
    full_vocab: list[str],
    full_merges: list[tuple[str, str]],
    protected_tokens: set[str],
    target_size: int = 98304,
) -> tuple[set[str], list[tuple[str, str]]]:
    """Executa a seleção gulosa de tokens com blindagem obrigatória e fechamento transitivo BPE.

    Blindagem obrigatória:
        - Todos os tokens em `protected_tokens` (256 bytes elementares e tokens especiais)
          são preservados incondicionalmente, junto com todo o grafo de seus ancestrais recursivos.

    Seleção gulosa:
        - Candidatos de `full_vocab` são ranqueados por contagem de ocorrência decrescente;
        - Para cada candidato, é computado o incremento transitivo de dependências:
          Delta(t) = ({t} U Ancestrais(t)) \\ S;
        - O token e seus ancestrais faltantes são adicionados se |S| + |Delta(t)| <= target_size;
        - Se a lista de frequências for percorrida e |S| < target_size, realiza preenchimento
          residual com os tokens de menor custo marginal transitivo até atingir rigorosamente target_size.

    Args:
        frequencies: Dicionário contendo a frequência de tokens observada no corpus de calibração.
        full_vocab: Lista ordenada de todos os tokens do vocabulário original (ex: Qwen3.5 248K).
        full_merges: Lista de regras de merge (left, right) do tokenizador original.
        protected_tokens: Conjunto de tokens obrigatórios que não podem ser podados.
        target_size: Quantidade exata de tokens no vocabulário final (padrão 98304).

    Returns:
        Tupla (selected_vocab, pruned_merges):
            - selected_vocab: Conjunto de exatamente `target_size` tokens fechado sob merges.
            - pruned_merges: Lista de regras de merge ativas cujos constituintes e resultado
              pertencem a selected_vocab, preservando a ordem original.

    Raises:
        ValueError: Se target_size for menor que os tokens protegidos (e seus ancestrais)
            ou maior que len(full_vocab).
        RuntimeError: Se não for possível atingir rigorosamente target_size.
    """
    if target_size > len(full_vocab):
        raise ValueError(
            f"target_size ({target_size}) é maior que o tamanho do vocabulário original ({len(full_vocab)})."
        )

    dep_graph = build_dependency_graph(full_merges)
    memo: dict[str, set[str]] = {}

    # Inicializa com conjunto protegido e todos os seus ancestrais transitivos
    selected_vocab: set[str] = set(protected_tokens)
    for token in list(protected_tokens):
        selected_vocab.update(compute_ancestors(token, dep_graph, memo))

    if len(selected_vocab) > target_size:
        raise ValueError(
            f"target_size ({target_size}) é menor que o conjunto obrigatório protegido "
            f"e seus ancestrais transitivos ({len(selected_vocab)})."
        )

    if len(selected_vocab) < target_size:
        # Fila de prioridade de candidatos: ordenação estável por frequência decrescente
        candidates = [t for t in full_vocab if t not in selected_vocab]
        candidates.sort(key=lambda t: frequencies.get(t, 0), reverse=True)

        for token in candidates:
            if len(selected_vocab) == target_size:
                break
            if token in selected_vocab:
                continue

            needed = (compute_ancestors(token, dep_graph, memo) | {token}) - selected_vocab
            if len(selected_vocab) + len(needed) <= target_size:
                selected_vocab.update(needed)

    # Preenchimento residual caso |S| < target_size
    if len(selected_vocab) < target_size:
        remaining = [t for t in full_vocab if t not in selected_vocab]
        while len(selected_vocab) < target_size and remaining:
            best_token = None
            best_needed: set[str] = set()
            best_cost = float("inf")

            for t in remaining:
                needed = (compute_ancestors(t, dep_graph, memo) | {t}) - selected_vocab
                cost = len(needed)
                if 0 < cost <= (target_size - len(selected_vocab)):
                    if cost < best_cost:
                        best_cost = cost
                        best_token = t
                        best_needed = needed
                        if cost == 1:
                            break

            if best_token is None:
                break

            selected_vocab.update(best_needed)
            remaining = [t for t in remaining if t not in selected_vocab]

    if len(selected_vocab) != target_size:
        raise RuntimeError(
            f"Não foi possível atingir o target_size exato de {target_size}; "
            f"selecionados {len(selected_vocab)} tokens."
        )

    # Filtragem das regras de merge: preserva apenas regras cujos operandos e resultado estão no vocabulário
    pruned_merges: list[tuple[str, str]] = []
    for left, right in full_merges:
        res = _merge_result(left, right)
        direct = left + right
        if (direct in selected_vocab or res in selected_vocab):
            if left in selected_vocab and right in selected_vocab:
                pruned_merges.append((left, right))

    return selected_vocab, pruned_merges
