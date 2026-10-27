"""Testes unitários para o grafo de dependências BPE e fechamento de ancestrais."""

from sore.data.prune_tokenizer import (
    build_dependency_graph,
    compute_ancestors,
    is_merge_closed,
)


def test_dependency_graph_and_ancestors():
    # Merges: ("a", "b") -> "ab", ("ab", "c") -> "abc"
    merges = [("a", "b"), ("ab", "c")]
    graph = build_dependency_graph(merges)
    assert graph["ab"] == ("a", "b")
    assert graph["abc"] == ("ab", "c")
    memo: dict[str, set[str]] = {}
    assert compute_ancestors("ab", graph, memo) == {"a", "b"}
    assert compute_ancestors("abc", graph, memo) == {"a", "b", "c", "ab"}
    assert is_merge_closed({"a", "b", "c", "ab", "abc"}, merges)
    assert not is_merge_closed({"a", "b", "abc"}, merges)  # falta "ab" e "c"


def test_empty_merges_and_base_tokens():
    graph = build_dependency_graph([])
    assert graph == {}
    memo: dict[str, set[str]] = {}
    assert compute_ancestors("base", graph, memo) == set()
    assert is_merge_closed(set(), [])
    assert is_merge_closed({"token1", "token2"}, [])


def test_multilevel_dag_closure():
    # DAG de 3 níveis:
    # ("a", "b") -> "ab"
    # ("c", "d") -> "cd"
    # ("ab", "cd") -> "abcd"
    merges = [("a", "b"), ("c", "d"), ("ab", "cd")]
    graph = build_dependency_graph(merges)

    memo: dict[str, set[str]] = {}
    ancestors_abcd = compute_ancestors("abcd", graph, memo)
    assert ancestors_abcd == {"a", "b", "c", "d", "ab", "cd"}

    # Falta "d"
    assert not is_merge_closed({"abcd", "ab", "cd", "a", "b", "c"}, merges)
    # Todos presentes
    assert is_merge_closed({"abcd", "ab", "cd", "a", "b", "c", "d"}, merges)
    # Apenas nós intermediários sem a raiz
    assert is_merge_closed({"ab", "a", "b"}, merges)


def test_memoization_avoids_recomputation():
    merges = [("x", "y"), ("xy", "z")]
    graph = build_dependency_graph(merges)
    memo: dict[str, set[str]] = {}

    anc1 = compute_ancestors("xyz", graph, memo)
    assert "xyz" in memo
    assert memo["xyz"] is anc1
    # Chamada repetida deve retornar o memoizado
    anc2 = compute_ancestors("xyz", graph, memo)
    assert anc2 is anc1
