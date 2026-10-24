"""Amostra real de texto PT para os portões de qualidade (A/B de perda em ~20M tokens).

O shard sintético (`scripts/make_synthetic_shard.py`) é aprendido quase por completo, então não
discrimina nada: os testes de qualidade precisam de texto de verdade. A amostra é tokenizada com o
tokenizer do Qwen3.5 (vocab ~248K) e reduzida provisoriamente para um vocabulário de 65.536 ids:
EOS = 0, UNK = 1 e os 65.534 ids mais frequentes da própria amostra viram 2, 3, ... em ordem de
frequência. O vocabulário definitivo do projeto será outro; aqui só precisa ser estável e estar
no formato que o modelo consome.

A lógica (remap, comprimentos, tokenização, duas passadas) fica aqui para ser testável offline; o
script `scripts/make_text_sample.py` só cuida da CLI e da rede. Downloads são dados não confiáveis:
só são lidos como gzip/JSON, nunca executados.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np

from sore.train.data import write_token_shard

EOS_ID = 0
UNK_ID = 1
SAMPLE_VOCAB = 65536
MAX_DOC_LENGTHS = 100_000  # tamanho do reservoir de comprimentos de documento
_FLUSH_TOKENS = 4_000_000  # tokens acumulados antes de gravar/contar (bincount por doc seria lento)
_REMAP_CHUNK = 16_000_000
_ENCODE_BATCH = 512
_SHA_RE = re.compile(r"[0-9a-f]{40}")


def build_remap(counts: np.ndarray, eos_old_id: int, vocab_size: int = SAMPLE_VOCAB) -> np.ndarray:
    """Tabela id antigo -> id novo (int64): EOS -> 0, top `vocab_size - 2` por frequência -> 2.., resto -> 1.

    O EOS não disputa vaga com os demais ids, e ids com contagem 0 nunca entram no top (viram UNK).
    O empate é desfeito pelo menor id antigo (ordenação estável sobre a contagem decrescente),
    para o resultado ser determinístico.
    """
    counts = np.asarray(counts)
    remap = np.full(len(counts), UNK_ID, dtype=np.int64)
    candidates = np.flatnonzero(counts > 0)
    candidates = candidates[candidates != eos_old_id]
    order = np.argsort(-counts[candidates], kind="stable")
    top = candidates[order][: vocab_size - 2]
    remap[top] = np.arange(2, 2 + len(top), dtype=np.int64)
    remap[eos_old_id] = EOS_ID
    return remap


def sample_doc_lengths(lengths: Sequence[int], total: int, rng: np.random.Generator) -> list[int]:
    """Sorteia comprimentos (com reposição) até somar exatamente `total`; o último é truncado.

    Serve para fabricar fronteiras de documento realistas em testes que usam tokens sem EOS.
    """
    pool = np.asarray(lengths, dtype=np.int64)
    if pool.size == 0 or pool.min() < 1:
        raise ValueError("lengths deve ser não vazio e conter só inteiros >= 1")
    out: list[int] = []
    acc = 0
    while acc < total:
        draw = rng.choice(pool, size=max(1, (total - acc) // int(pool.mean()) + 1))
        for n in draw.tolist():
            n = min(n, total - acc)
            out.append(n)
            acc += n
            if acc == total:
                break
    return out


def tokenize_docs(docs: Iterable[str], tokenizer, eos_old_id: int) -> Iterator[np.ndarray]:
    """Um array uint32 por documento (ids antigos) terminado em `eos_old_id`.

    Documentos que não geram nenhum token são descartados (não viram um "documento só de EOS").
    Usa `encode_batch`, que paraleliza em Rust; a ordem dos documentos é preservada.
    """
    batch: list[str] = []

    def flush() -> Iterator[np.ndarray]:
        for enc in tokenizer.encode_batch(batch, add_special_tokens=False):
            if not enc.ids:
                continue
            arr = np.empty(len(enc.ids) + 1, dtype=np.uint32)
            arr[:-1] = enc.ids
            arr[-1] = eos_old_id
            yield arr

    for text in docs:
        batch.append(text)
        if len(batch) >= _ENCODE_BATCH:
            yield from flush()
            batch = []
    if batch:
        yield from flush()


def iter_jsonl_gz_texts(path: Path, field: str = "text") -> Iterator[str]:
    """Lê um `.json.gz` JSON-lines em streaming e devolve o campo `field` de cada linha."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)[field]


class LengthReservoir:
    """Reservoir sampling (algoritmo R): amostra uniforme de até `k` comprimentos de um fluxo."""

    def __init__(self, k: int, rng: np.random.Generator):
        self.k, self._rng, self._seen = k, rng, 0
        self.items: list[int] = []

    def add(self, value: int) -> None:
        self._seen += 1
        if len(self.items) < self.k:
            self.items.append(value)
        else:
            j = int(self._rng.integers(0, self._seen))
            if j < self.k:
                self.items[j] = value


def generate_sample(
    texts: Iterable[str],
    tokenizer,
    eos_old_id: int,
    out_dir: Path,
    n_tokens: int,
    seed: int = 0,
    vocab_size: int = SAMPLE_VOCAB,
) -> dict:
    """Duas passadas: tokeniza (ids antigos) para um temporário contando frequências; remapeia e grava.

    Escreve `sample_000.bin`(+`.json`), `remap.npy` e `doc_lengths.json` em `out_dir` e devolve
    `{"n_tokens", "n_docs", "unk_rate"}`. O shard tem exatamente `n_tokens` tokens e termina em EOS:
    o último documento é truncado se preciso (o último token vira EOS). Levanta `RuntimeError`
    se `texts` acabar antes de `n_tokens`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "tokens_old.tmp"
    n_vocab_old = tokenizer.get_vocab_size(with_added_tokens=True)
    counts = np.zeros(n_vocab_old, dtype=np.int64)
    reservoir = LengthReservoir(MAX_DOC_LENGTHS, np.random.default_rng(seed))
    n_docs = written = 0
    pending: list[np.ndarray] = []
    pending_n = 0

    def flush(f) -> None:
        nonlocal pending, pending_n
        if not pending:
            return
        chunk = np.concatenate(pending)
        counts[:] += np.bincount(chunk, minlength=n_vocab_old)
        f.write(chunk.astype("<u4").tobytes())
        pending, pending_n = [], 0

    with open(tmp, "wb") as f:
        for arr in tokenize_docs(texts, tokenizer, eos_old_id):
            room = n_tokens - written
            if len(arr) > room:
                arr = arr[:room].copy()
                arr[-1] = eos_old_id
            pending.append(arr)
            pending_n += len(arr)
            written += len(arr)
            n_docs += 1
            reservoir.add(len(arr))
            if pending_n >= _FLUSH_TOKENS:
                flush(f)
            if written >= n_tokens:
                break
        flush(f)
    if written < n_tokens:
        tmp.unlink()
        raise RuntimeError(f"fonte esgotada: {written} < {n_tokens} tokens")

    remap = build_remap(counts, eos_old_id, vocab_size)
    old = np.memmap(tmp, dtype="<u4", mode="r", shape=(n_tokens,))
    new = np.empty(n_tokens, dtype=np.uint32)
    n_unk = 0
    for lo in range(0, n_tokens, _REMAP_CHUNK):
        mapped = remap[old[lo : lo + _REMAP_CHUNK]]
        n_unk += int((mapped == UNK_ID).sum())
        new[lo : lo + len(mapped)] = mapped
    del old
    write_token_shard(out_dir / "sample_000.bin", new)
    np.save(out_dir / "remap.npy", remap)
    (out_dir / "doc_lengths.json").write_text(json.dumps({"lengths": reservoir.items}))
    tmp.unlink()
    return {"n_tokens": n_tokens, "n_docs": n_docs, "unk_rate": n_unk / n_tokens}


def resolve_revision(repo: str, revision: str, kind: str = "models") -> str:
    """Resolve `revision` para o sha do commit via API do HF Hub.

    Um sha de 40 hex já é imutável e passa direto, sem rede. Qualquer outra referência ("main", tag)
    exige a API: se ela falhar, levanta `RuntimeError` em vez de cair de volta na string recebida,
    pois um "main" gravado no meta.json (e na chave do cache dos brutos) quebraria a reprodutibilidade
    sem aviso.
    """
    if _SHA_RE.fullmatch(revision):
        return revision
    url = f"https://huggingface.co/api/{kind}/{repo}/revision/{revision}"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)["sha"]
    except Exception as e:
        raise RuntimeError(
            f"não foi possível resolver {repo}@{revision} para um sha ({url}): {e!r}. "
            "Passe --revision com o sha de 40 caracteres ou restabeleça a rede."
        ) from e


def download(url: str, dest: Path) -> Path:
    """Baixa `url` para `dest` (via `.part` + rename, para nunca deixar arquivo pela metade). Reaproveita se existir."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
        while block := r.read(1 << 20):
            f.write(block)
    os.replace(part, dest)
    return dest


def iter_remote_texts(urls: Iterable[str], raw_dir: Path, on_file: Callable[[str], None]) -> Iterator[str]:
    """Baixa cada `.json.gz` em `raw_dir` (um por vez, sob demanda) e produz seus textos."""
    for url in urls:
        path = download(url, Path(raw_dir) / url.rsplit("/", 1)[-1])
        on_file(url)
        yield from iter_jsonl_gz_texts(path)
