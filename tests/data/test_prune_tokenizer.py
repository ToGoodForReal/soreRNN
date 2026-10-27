"""Testes unitários para o grafo de dependências BPE e fechamento de ancestrais."""

from sore.data.prune_tokenizer import (
    build_dependency_graph,
    build_remapping_tensors,
    compute_ancestors,
    export_pruned_tokenizer,
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


def test_build_remapping_tensors():
    import torch

    old_vocab = {"<0x00>": 0, "a": 1, "b": 2, "unused": 3, "<|endoftext|>": 4}
    new_vocab = {"<0x00>": 0, "a": 1, "b": 2, "<|endoftext|>": 3}
    new_to_old, old_to_new = build_remapping_tensors(old_vocab, new_vocab)

    assert new_to_old.shape == (4,)
    assert old_to_new.shape == (5,)
    assert new_to_old.tolist() == [0, 1, 2, 4]
    assert old_to_new[0] == 0 and old_to_new[1] == 1 and old_to_new[2] == 2
    assert old_to_new[3] == -1  # descarte
    assert old_to_new[4] == 3
    # Invariante de destilação
    for new_id in range(len(new_vocab)):
        assert old_to_new[new_to_old[new_id]] == new_id


def test_export_pruned_tokenizer(tmp_path):
    import json
    import torch
    from tokenizers import Tokenizer

    # Criar um tokenizer fonte sintético no formato HuggingFace
    bytes_tokens = {f"<0x{i:02X}>": i for i in range(256)}
    # Adicionar tokens compostos e especiais
    vocab = dict(bytes_tokens)
    vocab["a"] = 256
    vocab["b"] = 257
    vocab["c"] = 258
    vocab["ab"] = 259
    vocab["abc"] = 260
    vocab["unused_tok"] = 261
    vocab["<|endoftext|>"] = 262

    merges = [("a", "b"), ("ab", "c")]
    merges_str = [f"{l} {r}" for l, r in merges]

    tokenizer_data = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 262,
                "content": "<|endoftext|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        ],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges_str,
        },
    }

    source_dir = tmp_path / "source_tok"
    source_dir.mkdir()
    with open(source_dir / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f)
    with open(source_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump({"eos_token": "<|endoftext|>", "vocab_size": 263}, f)
    with open(source_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump({"eos_token": "<|endoftext|>"}, f)

    # Frequências: preferir "abc" (e transitivamente "ab", "a", "b", "c") sobre "unused_tok"
    freqs_path = tmp_path / "freqs.json"
    with open(freqs_path, "w", encoding="utf-8") as f:
        json.dump({"abc": 100, "unused_tok": 1}, f)

    out_dir = tmp_path / "pruned_tok"
    target_size = 262  # 256 bytes + "a" + "b" + "c" + "ab" + "abc" + "<|endoftext|>" = 262 tokens (elimina unused_tok)

    export_pruned_tokenizer(
        source_tokenizer_path=str(source_dir),
        output_dir=str(out_dir),
        target_size=target_size,
        frequencies_path=str(freqs_path),
    )

    # Verificar existência de todos os arquivos
    assert (out_dir / "tokenizer.json").exists()
    assert (out_dir / "tokenizer_config.json").exists()
    assert (out_dir / "special_tokens_map.json").exists()
    assert (out_dir / "remapping.pt").exists()

    # Verificar carregamento pelo tokenizers HF
    reloaded = Tokenizer.from_file(str(out_dir / "tokenizer.json"))
    assert reloaded.get_vocab_size(with_added_tokens=True) == target_size

    # Verificar remapping.pt
    remap = torch.load(out_dir / "remapping.pt")
    new_to_old = remap["new_to_old"]
    old_to_new = remap["old_to_new"]
    assert new_to_old.shape == (target_size,)
    assert old_to_new.shape == (263,)
    assert old_to_new[vocab["unused_tok"]] == -1  # descartado

    # Invariante de destilação
    for new_id in range(target_size):
        assert old_to_new[new_to_old[new_id]] == new_id

    # <|endoftext|> deve estar na posição final (target_size - 1)
    assert reloaded.token_to_id("<|endoftext|>") == target_size - 1
    assert old_to_new[vocab["<|endoftext|>"]] == target_size - 1


def test_prune_tokenizer_cli(tmp_path):
    import json
    from scripts.prune_tokenizer import main as cli_main

    # Criar tokenizer fonte sintético
    bytes_tokens = {f"<0x{i:02X}>": i for i in range(256)}
    vocab = dict(bytes_tokens)
    vocab["extra"] = 256
    vocab["<|endoftext|>"] = 257

    tokenizer_data = {
        "version": "1.0",
        "added_tokens": [
            {
                "id": 257,
                "content": "<|endoftext|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        ],
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": [],
        },
    }

    source_dir = tmp_path / "src"
    source_dir.mkdir()
    with open(source_dir / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f)

    out_dir = tmp_path / "dst"
    cli_main(["--source", str(source_dir), "--out", str(out_dir), "--target-size", "257"])

    assert (out_dir / "tokenizer.json").exists()
    assert (out_dir / "remapping.pt").exists()


def test_export_pruned_tokenizer_missing_frequencies_file(tmp_path):
    import pytest
    import json

    # Configuração mínima
    bytes_tokens = {f"<0x{i:02X}>": i for i in range(256)}
    vocab = dict(bytes_tokens)
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    with open(source_dir / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump({"version": "1.0", "model": {"type": "BPE", "vocab": vocab, "merges": []}}, f)

    with pytest.raises(FileNotFoundError, match="Arquivo de frequências não encontrado"):
        export_pruned_tokenizer(
            source_tokenizer_path=str(source_dir),
            output_dir=str(tmp_path / "dst"),
            target_size=256,
            frequencies_path=str(tmp_path / "inexistente.json"),
        )


def test_export_pruned_tokenizer_unsupported_extension(tmp_path):
    import pytest
    import json

    bytes_tokens = {f"<0x{i:02X}>": i for i in range(256)}
    vocab = dict(bytes_tokens)
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    with open(source_dir / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump({"version": "1.0", "model": {"type": "BPE", "vocab": vocab, "merges": []}}, f)

    bad_freqs = tmp_path / "freqs.txt"
    bad_freqs.write_text("dummy", encoding="utf-8")

    with pytest.raises(ValueError, match="Extensão de arquivo de frequências não suportada"):
        export_pruned_tokenizer(
            source_tokenizer_path=str(source_dir),
            output_dir=str(tmp_path / "dst"),
            target_size=256,
            frequencies_path=str(bad_freqs),
        )


def test_export_pruned_tokenizer_pt_dict_frequencies(tmp_path):
    import json
    import torch
    from tokenizers import Tokenizer

    bytes_tokens = {f"<0x{i:02X}>": i for i in range(256)}
    vocab = dict(bytes_tokens)
    vocab["a"] = 256
    vocab["b"] = 257
    vocab["c"] = 258
    vocab["ab"] = 259
    vocab["abc"] = 260
    vocab["unused_tok"] = 261
    vocab["<|endoftext|>"] = 262

    merges = [("a", "b"), ("ab", "c")]
    merges_str = [f"{l} {r}" for l, r in merges]

    tokenizer_data = {
        "version": "1.0",
        "added_tokens": [
            {
                "id": 262,
                "content": "<|endoftext|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        ],
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": merges_str,
        },
    }

    source_dir = tmp_path / "src"
    source_dir.mkdir()
    with open(source_dir / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f)

    # Salva frequências em .pt com chaves int (IDs originais)
    pt_freqs_path = tmp_path / "freqs.pt"
    # vocab["abc"] = 260, vocab["unused_tok"] = 261
    torch.save({260: 100, 261: 1}, pt_freqs_path)

    out_dir = tmp_path / "dst"
    target_size = 262  # 256 bytes + 5 (a, b, c, ab, abc) + 1 (<|endoftext|>)

    export_pruned_tokenizer(
        source_tokenizer_path=str(source_dir),
        output_dir=str(out_dir),
        target_size=target_size,
        frequencies_path=str(pt_freqs_path),
    )

    reloaded = Tokenizer.from_file(str(out_dir / "tokenizer.json"))
    assert reloaded.get_vocab_size(with_added_tokens=True) == target_size
    remap = torch.load(out_dir / "remapping.pt")
    old_to_new = remap["old_to_new"]
    # "abc" foi mantido e "unused_tok" foi descartado (-1)
    assert old_to_new[vocab["abc"]] != -1
    assert old_to_new[vocab["unused_tok"]] == -1





