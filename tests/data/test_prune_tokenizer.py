"""Testes unitários para o grafo de dependências BPE e fechamento de ancestrais."""

from sore.data.prune_tokenizer import (
    build_dependency_graph,
    compute_ancestors,
    is_merge_closed,
    select_pruned_vocab,
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


def test_select_pruned_vocab_exact_size_and_closure():
    # Vocabulario sintético com bytes, especiais e compostos
    bytes_tokens = {f"<0x{i:02X}>" for i in range(256)}
    special_tokens = {"<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"}
    protected = bytes_tokens | special_tokens

    # Merges: ("<0x61>", "<0x62>") -> "ab", ("ab", "<0x63>") -> "abc", ("<0x64>", "<0x65>") -> "de"
    full_merges = [("<0x61>", "<0x62>"), ("ab", "<0x63>"), ("<0x64>", "<0x65>")]
    full_vocab = list(protected) + ["ab", "abc", "de"]
    freqs = {"abc": 100, "de": 50, "ab": 10}

    target_size = len(protected) + 2  # Cabe protected + "ab" e "abc" (custo total 2), mas não "de"
    selected_vocab, pruned_merges = select_pruned_vocab(
        freqs, full_vocab, full_merges, protected, target_size=target_size
    )
    assert len(selected_vocab) == target_size
    assert protected.issubset(selected_vocab)
    assert "ab" in selected_vocab and "abc" in selected_vocab
    assert is_merge_closed(selected_vocab, pruned_merges)


def test_select_pruned_vocab_invalid_target_size():
    import pytest

    protected = {"<0x00>", "<0x01>"}
    full_vocab = ["<0x00>", "<0x01>", "a"]
    # target_size menor que conjunto protegido
    with pytest.raises(ValueError, match="menor"):
        select_pruned_vocab({}, full_vocab, [], protected, target_size=1)
    # target_size maior que vocabulário completo
    with pytest.raises(ValueError, match="maior"):
        select_pruned_vocab({}, full_vocab, [], protected, target_size=10)


def test_select_pruned_vocab_residual_filling():
    protected = {"<0x00>", "<0x01>"}
    # Merges: ("<0x00>", "<0x01>") -> "x", ("x", "<0x00>") -> "xx"
    merges = [("<0x00>", "<0x01>"), ("x", "<0x00>")]
    full_vocab = ["<0x00>", "<0x01>", "x", "xx", "y"]
    # Sem frequências para "x", apenas "y" tem frequência (>0).
    # O loop guloso consome "y", ocupando 3 slots (protected + "y").
    # O 4º slot obrigatoriamente vem do preenchimento residual,
    # que deve escolher "x" (custo transitivo 1) em vez de "xx" (custo transitivo 2).
    freqs = {"y": 10}
    selected_vocab, pruned_merges = select_pruned_vocab(
        freqs, full_vocab, merges, protected, target_size=4
    )
    assert len(selected_vocab) == 4
    assert protected.issubset(selected_vocab)
    assert "y" in selected_vocab
    assert "x" in selected_vocab
    assert "xx" not in selected_vocab
    assert is_merge_closed(selected_vocab, pruned_merges)


def test_select_pruned_vocab_zero_frequencies_use_residual_filling():
    protected = {"<0x00>"}
    # Merges: ("<0x00>", "<0x00>") -> "a", ("a", "<0x00>") -> "b"
    merges = [("<0x00>", "<0x00>"), ("a", "<0x00>")]
    full_vocab = ["<0x00>", "a", "b"]
    # Frequências com 0 para "b" e sem frequência para "a"
    # "b" tem custo 2 (precisa de "a"). O loop guloso ignora freq <= 0.
    # O preenchimento residual escolhe "a" (menor custo transitivo = 1).
    freqs = {"b": 0}
    selected_vocab, pruned_merges = select_pruned_vocab(
        freqs, full_vocab, merges, protected, target_size=2
    )
    assert len(selected_vocab) == 2
    assert "a" in selected_vocab
    assert "b" not in selected_vocab
    assert is_merge_closed(selected_vocab, pruned_merges)


