"""Poda do tokenizer Qwen3.5 (BPE byte-level, 248K ids) para o vocabulário do modelo (spec §4).

O conjunto mantido é fechado sob os merges: todo token mantido tem os dois "pais" do merge que o
cria também mantidos, até chegar aos 256 tokens de byte (as raízes, que nunca saem). Assim, todo
token preservado continua alcançável e todo texto continua codificável (fallback para bytes, sem
UNK). Os merges mantidos são os cujo resultado ficou, na ordem (rank) original, e os tokens
adicionados (EOS, chat, tool, think, visão, FIM) são preservados todos.

A escolha é gulosa por frequência num corpus amostrado: em ordem decrescente de contagem, cada
token entra junto com os ancestrais que ainda faltam, se couber no orçamento. Os ids novos seguem a
ordem dos ids antigos (vocabulário BPE primeiro, tokens adicionados no fim), e `old_to_new` (-1 para
os podados) permite restringir os logits de um teacher ao subvocabulário (destilação).
"""
from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from typing import Any

import numpy as np


def _merge_pair(merge: str | list[str]) -> tuple[str, str]:
    """Merges vêm como "a b" (formato antigo) ou ["a", "b"] (tokenizers >= 0.20)."""
    if isinstance(merge, str):
        a, b = merge.split(" ", 1)
        return a, b
    return merge[0], merge[1]


def merge_parents(tok_json: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Token -> os dois tokens cujo merge o cria (o de menor rank, se houver mais de um)."""
    parents: dict[str, tuple[str, str]] = {}
    for merge in tok_json["model"]["merges"]:
        a, b = _merge_pair(merge)
        parents.setdefault(a + b, (a, b))
    return parents


def count_tokens(texts: Iterable[str], tokenizer, n_ids: int, batch: int = 1024) -> np.ndarray:
    """Contagem de cada id em `texts` (sem tokens especiais), `int64` de tamanho `n_ids`."""
    counts = np.zeros(n_ids, dtype=np.int64)
    buf: list[str] = []

    def flush() -> None:
        encs = tokenizer.encode_batch(buf, add_special_tokens=False)
        ids = np.fromiter((i for e in encs for i in e.ids), dtype=np.int64)
        counts[:] += np.bincount(ids, minlength=n_ids)

    for text in texts:
        buf.append(text)
        if len(buf) >= batch:
            flush()
            buf = []
    if buf:
        flush()
    return counts


def select_vocab(tok_json: dict[str, Any], counts: np.ndarray, budget: int) -> set[str]:
    """Tokens BPE mantidos (raízes + escolha gulosa fechada sob merges), no máximo `budget`.

    `budget` conta só o vocabulário BPE (os tokens adicionados ficam de fora). ValueError se o
    orçamento não comportar nem as raízes.
    """
    vocab: dict[str, int] = tok_json["model"]["vocab"]
    parents = merge_parents(tok_json)
    roots = {t for t in vocab if t not in parents}
    if len(roots) > budget:
        raise ValueError(f"orçamento {budget} menor que as {len(roots)} raízes (bytes)")
    keep = set(roots)
    by_id = sorted(vocab.items(), key=lambda kv: kv[1])
    tokens = np.array([t for t, _ in by_id], dtype=object)
    ids = np.array([i for _, i in by_id], dtype=np.int64)
    c = counts[ids]
    order = np.lexsort((ids, -c))  # contagem decrescente, empate pelo menor id antigo
    for k in order:
        if len(keep) >= budget or c[k] == 0:
            break
        need = _missing_closure(tokens[k], parents, keep)
        if len(keep) + len(need) <= budget:
            keep |= need
    return keep


def _missing_closure(token: str, parents: dict[str, tuple[str, str]], keep: set[str]) -> set[str]:
    need: set[str] = set()
    stack = [token]
    while stack:
        t = stack.pop()
        if t in keep or t in need:
            continue
        need.add(t)
        stack.extend(parents.get(t, ()))
    return need


def build_pruned(tok_json: dict[str, Any], keep: set[str]) -> tuple[dict[str, Any], np.ndarray]:
    """O tokenizer podado (JSON do `tokenizers`) e o `old_to_new` (int64, -1 = podado).

    `keep` precisa ser fechado sob merges (ValueError caso contrário).
    """
    model = tok_json["model"]
    vocab: dict[str, int] = model["vocab"]
    parents = merge_parents(tok_json)
    for t in keep:
        if t not in vocab:
            raise ValueError(f"token {t!r} fora do vocabulário")
        if any(p not in keep for p in parents.get(t, ())):
            raise ValueError(f"conjunto não é fechado sob merges: falta um pai de {t!r}")

    added = sorted(tok_json["added_tokens"], key=lambda a: a["id"])
    n_old = max([*vocab.values(), *(a["id"] for a in added)]) + 1
    old_to_new = np.full(n_old, -1, dtype=np.int64)
    kept = sorted(keep, key=vocab.__getitem__)
    new_vocab = {t: i for i, t in enumerate(kept)}
    old_to_new[[vocab[t] for t in kept]] = np.arange(len(kept))
    new_added = []
    for j, a in enumerate(added):
        new_id = len(kept) + j
        old_to_new[a["id"]] = new_id
        new_added.append({**a, "id": new_id})

    out = copy.deepcopy({k: v for k, v in tok_json.items() if k != "model"})
    out["model"] = {**{k: v for k, v in model.items() if k not in ("vocab", "merges")}, "vocab": new_vocab}
    out["model"]["merges"] = [m for m in model["merges"] if "".join(_merge_pair(m)) in keep]
    out["added_tokens"] = new_added
    return out, old_to_new


def without_added_tokens(tok_json: dict[str, Any]) -> dict[str, Any]:
    """Cópia sem os tokens adicionados, para tokenizar texto cru do corpus.

    Com eles, uma string literal como "<|im_start|>" num documento da web viraria o token especial.
    Sem eles, ela vira bytes/BPE comuns; o EOS entre documentos é posto à mão pelo gerador.
    """
    out = copy.deepcopy(tok_json)
    out["added_tokens"] = []
    return out


def to_json_str(tok_json: dict[str, Any]) -> str:
    return json.dumps(tok_json, ensure_ascii=False)
