"""Testes da poda do tokenizer (`sore.train.tokenizer_prune`) sobre um BPE byte-level pequeno."""
import json
import unicodedata

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from sore.train.tokenizer_prune import (
    build_pruned,
    count_tokens,
    merge_parents,
    select_vocab,
    to_json_str,
    without_added_tokens,
)

CORPUS = [
    "o gato come o peixe e o cachorro come a ração",
    "a programação em python é divertida: def soma(a, b): return a + b",
    "ações, informações e situações são palavras com cedilha e til",
    "The quick brown fox jumps over the lazy dog 12345",
    '{"nome": "Ana", "idade": 31, "cidades": ["Recife", "Natal"]}',
] * 40
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]


@pytest.fixture(scope="module")
def tok_json():
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=700, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False
    )
    tok.train_from_iterator(CORPUS, trainer)
    tok.add_special_tokens(SPECIALS)
    return json.loads(tok.to_str())


def _n_ids(tj):
    return max([*tj["model"]["vocab"].values(), *(a["id"] for a in tj["added_tokens"])]) + 1


def _prune(tj, budget):
    full = Tokenizer.from_str(to_json_str(tj))
    counts = count_tokens(CORPUS, full, _n_ids(tj))
    keep = select_vocab(tj, counts, budget)
    return full, keep, *build_pruned(tj, keep)


def test_select_vocab_is_closed_under_merges_and_within_budget(tok_json):
    _, keep, _, _ = _prune(tok_json, 320)
    parents = merge_parents(tok_json)
    assert len(keep) <= 320
    assert all(p in keep for t in keep for p in parents.get(t, ()))
    roots = {t for t in tok_json["model"]["vocab"] if t not in parents}
    assert len(roots) == 256 and roots <= keep


def test_pruned_roundtrip_old_to_new_and_specials(tok_json):
    full, keep, pruned_json, old_to_new = _prune(tok_json, 320)
    pruned = Tokenizer.from_str(to_json_str(pruned_json))
    assert pruned.get_vocab_size() == len(keep) + len(SPECIALS)
    texts = CORPUS[:5] + ["texto inédito: çãõ€ 🙂 \t\n ZZZ qwerty"]
    for text in texts:
        ids = pruned.encode(text, add_special_tokens=False).ids
        assert pruned.decode(ids) == unicodedata.normalize("NFC", text)  # cobertura total, sem UNK
        # todo id antigo mantido mapeia para o id novo do mesmo token
    for tok, old in tok_json["model"]["vocab"].items():
        new = old_to_new[old]
        assert (new >= 0) == (tok in keep)
        if new >= 0:
            assert pruned.id_to_token(int(new)) == tok
    for s in SPECIALS:
        assert pruned.token_to_id(s) == old_to_new[full.token_to_id(s)]
    assert sorted(pruned.token_to_id(s) for s in SPECIALS) == list(range(len(keep), len(keep) + len(SPECIALS)))


def test_pruned_keeps_merge_order_and_frequent_tokens_unchanged(tok_json):
    full, keep, pruned_json, old_to_new = _prune(tok_json, 400)
    pair = lambda m: tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m)
    old = [pair(m) for m in tok_json["model"]["merges"]]
    new = [pair(m) for m in pruned_json["model"]["merges"]]
    assert new == [m for m in old if m[0] + m[1] in keep]  # subsequência, mesma ordem
    pruned = Tokenizer.from_str(to_json_str(pruned_json))
    text = CORPUS[0]
    full_ids = full.encode(text).ids
    if all(old_to_new[i] >= 0 for i in full_ids):  # todos os tokens sobreviveram: segmentação idêntica
        assert pruned.encode(text).ids == old_to_new[full_ids].tolist()
    assert len(pruned.encode(text).ids) <= len(full.encode(text).ids) * 1.5


def test_select_vocab_budget_below_roots_raises(tok_json):
    counts = np.ones(_n_ids(tok_json), dtype=np.int64)
    with pytest.raises(ValueError, match="raízes"):
        select_vocab(tok_json, counts, 100)


def test_build_pruned_rejects_unclosed_set(tok_json):
    parents = merge_parents(tok_json)
    child = next(t for t in tok_json["model"]["vocab"] if t in parents and parents[t][0] in parents)
    roots = {t for t in tok_json["model"]["vocab"] if t not in parents}
    with pytest.raises(ValueError, match="fechado"):
        build_pruned(tok_json, roots | {child})


def test_without_added_tokens_does_not_parse_literal_specials(tok_json):
    _, _, pruned_json, _ = _prune(tok_json, 320)
    with_specials = Tokenizer.from_str(to_json_str(pruned_json))
    plain = Tokenizer.from_str(to_json_str(without_added_tokens(pruned_json)))
    text = "veja <|im_start|> no meio do texto"
    special_id = with_specials.token_to_id("<|im_start|>")
    assert special_id in with_specials.encode(text).ids
    ids = plain.encode(text).ids
    assert special_id not in ids and max(ids) < special_id
    assert plain.decode(ids) == text
