"""Testes (CPU, offline) da amostra real de texto PT: remap de vocabulário, comprimentos e shard."""
from __future__ import annotations

import json

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from sore.train.data import PackedTokenStream, write_token_shard
from sore.train.text_sample import (
    EOS_ID,
    UNK_ID,
    build_remap,
    generate_sample,
    resolve_revision,
    sample_doc_lengths,
    tokenize_docs,
)


def test_build_remap_eos_top_k_and_unk():
    counts = np.array([5, 0, 9, 9, 1, 7])  # id antigo 3 é o EOS
    remap = build_remap(counts, eos_old_id=3, vocab_size=4)
    assert remap.dtype == np.int64 and len(remap) == 6
    assert remap[3] == EOS_ID == 0
    assert remap[2] == 2 and remap[5] == 3  # 9 (id 2) > 7 (id 5); o EOS não ocupa vaga
    assert set(remap[[0, 1, 4]]) == {UNK_ID}  # fora do top-2, e contagem 0


def test_build_remap_tie_breaks_by_lower_old_id():
    remap = build_remap(np.array([4, 4, 4, 1]), eos_old_id=3, vocab_size=4)
    assert remap[0] == 2 and remap[1] == 3 and remap[2] == UNK_ID


def test_sample_doc_lengths_sums_exactly():
    out = sample_doc_lengths([3, 50, 700], total=10_000, rng=np.random.default_rng(0))
    assert sum(out) == 10_000 and min(out) >= 1


def test_sample_end_to_end(tmp_path):
    vocab = {"[UNK]": 0, "<|endoftext|>": 1, "o": 2, "gato": 3, "come": 4, "peixe": 5, "cao": 6}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    eos_old = tok.token_to_id("<|endoftext|>")
    assert eos_old == 1

    docs = ["o gato come peixe", "o cao come", "gato peixe zebra o"]  # "zebra" cai em [UNK]
    arrays = list(tokenize_docs(iter(docs), tok, eos_old))
    assert len(arrays) == 3
    assert all(a.dtype == np.uint32 and a[-1] == eos_old for a in arrays)
    assert [len(a) for a in arrays] == [5, 4, 5]

    old = np.concatenate(arrays)
    counts = np.bincount(old, minlength=tok.get_vocab_size())
    remap = build_remap(counts, eos_old_id=eos_old)
    new = remap[old].astype(np.uint32)

    shard = tmp_path / "s.bin"
    write_token_shard(shard, new)
    got = np.fromfile(shard, dtype="<u4")
    ends = np.cumsum([len(a) for a in arrays]) - 1
    assert (got[ends] == 0).all()  # cada documento termina em EOS remapeado
    assert (np.flatnonzero(got == 0) == ends).all()  # e o EOS não aparece em nenhum outro lugar

    stream = PackedTokenStream([shard], seq_len=4, micro_batch=2, eos_id=0, seed=0)
    batch = next(iter(stream))
    assert batch is not None


def test_resolve_revision_failure_is_not_silent(monkeypatch):
    def boom(*a, **k):
        raise OSError("sem rede")

    monkeypatch.setattr("sore.train.text_sample.urllib.request.urlopen", boom)
    with pytest.raises(RuntimeError, match="Qwen/Qwen3.5-0.8B@main"):
        resolve_revision("Qwen/Qwen3.5-0.8B", "main")


def test_resolve_revision_full_sha_skips_the_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("não deveria chamar a rede")

    monkeypatch.setattr("sore.train.text_sample.urllib.request.urlopen", boom)
    sha = "2fc06364715b967f1860aea9cf38778875588b17"
    assert resolve_revision("Qwen/Qwen3.5-0.8B", sha) == sha


def test_generate_sample_truncates_exactly_and_ends_in_eos(tmp_path):
    vocab = {"[UNK]": 0, "<|endoftext|>": 1, "o": 2, "gato": 3, "come": 4}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    texts = ["o gato come"] * 10  # 4 tokens por documento (com EOS)

    stats = generate_sample(iter(texts), tok, 1, tmp_path, n_tokens=10, seed=0)

    shard = np.fromfile(tmp_path / "sample_000.bin", dtype="<u4")
    assert len(shard) == stats["n_tokens"] == 10 and stats["n_docs"] == 3
    assert shard[-1] == 0 and stats["unk_rate"] == 0.0
    assert json.loads((tmp_path / "sample_000.json").read_text())["n_tokens"] == 10
    assert sorted(json.loads((tmp_path / "doc_lengths.json").read_text())["lengths"]) == [2, 4, 4]
    assert not (tmp_path / "tokens_old.tmp").exists()
    assert len(np.load(tmp_path / "remap.npy")) == tok.get_vocab_size()
