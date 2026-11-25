"""Gerador dos shards de pré-treino: fontes do Hub -> tokenizer podado -> shards no disco (spec §5).

Cada fonte de `MIXTURE` recebe uma cota de `peso * total_tokens`. Os arquivos da fonte são lidos em
ordem e cada um vira um shard `<out>/<fonte>/<fonte>_<NNNNN>.bin` (+ sidecar), com cada documento
seguido do EOS. A cota é cortada na fronteira de documento; uma fonte que acaba antes da cota fica
com o que tiver (nada é repetido aqui).

Retomável: o shard é escrito num `.tmp` e só ganha o nome final (com o sidecar) quando o arquivo de
entrada termina; um shard pronto é pulado na retomada, um `.tmp` órfão é refeito do zero. O estado
de cada fonte (tokens, arquivo concluído) fica em `<out>/<fonte>/progress.json`.

Dedup entre fontes sobrepostas: o FineWeb2-HQ é um subconjunto do FineWeb-2, então os ids dos
documentos do HQ vão para `<shard>.ids` e o FineWeb-2 pula esses ids.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sore.train.data import SHARD_DTYPES

_ENCODE_DOCS = 256  # documentos por encode_batch (paralelo em Rust)
_PREFETCH = 8  # lotes de documentos lidos à frente pela thread de rede

# fonte -> fonte cujos ids de documento ela deve pular (processada antes)
DEDUP_AGAINST = {"fineweb2_pt": "fineweb2_hq_pt"}
ORDER_FIRST = ("fineweb2_hq_pt",)


@dataclass
class SourcePlan:
    name: str
    quota: int
    files: list[str]  # identificadores dos arquivos de entrada, na ordem de leitura


def plan_order(names: list[str]) -> list[str]:
    """As fontes de `ORDER_FIRST` primeiro (para a dedup), depois as demais na ordem dada."""
    return [n for n in ORDER_FIRST if n in names] + [n for n in names if n not in ORDER_FIRST]


def _prefetch(it: Iterator, size: int) -> Iterator:
    """Consome `it` numa thread (a leitura de rede corre enquanto a principal tokeniza)."""
    q: queue.Queue = queue.Queue(maxsize=size)
    done = object()

    def work() -> None:
        try:
            for item in it:
                q.put(item)
            q.put(done)
        except BaseException as e:  # repassa o erro para a thread principal
            q.put(e)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is done:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def _batches(docs: Iterator[tuple[str | None, str]], n: int) -> Iterator[list[tuple[str | None, str]]]:
    batch: list = []
    for doc in docs:
        batch.append(doc)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_progress(path: Path) -> dict:
    prog = json.loads(path.read_text()) if path.exists() else {"tokens": 0, "files_done": [], "exhausted": False}
    prog.setdefault("complete", False)  # progress.json antigo, sem o campo
    return prog


def _save_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def write_source(
    plan: SourcePlan,
    out_dir: Path,
    tokenizer,
    eos_id: int,
    dtype: str,
    read_docs: Callable[[str], Iterator[tuple[str | None, str]]],
    skip_ids: set[str] | None = None,
    save_ids: bool = False,
    log: Callable[[str], None] = print,
) -> int:
    """Gera (ou retoma) os shards de uma fonte até a cota; devolve o total de tokens da fonte.

    `read_docs(arquivo)` produz `(id, texto)`; `skip_ids` são ids a pular (dedup); com `save_ids`,
    os ids dos documentos gravados vão para `<shard>.ids` (um por linha).
    """
    np_dtype = SHARD_DTYPES[dtype]
    limit = np.iinfo(np_dtype).max
    src_dir = out_dir / plan.name
    src_dir.mkdir(parents=True, exist_ok=True)
    prog_path = src_dir / "progress.json"
    prog = _load_progress(prog_path)
    for i, file in enumerate(plan.files):
        # `complete`: a cota encheu na fronteira de documento (pode faltar um punhado de tokens);
        # sem ele, a retomada abriria mais um arquivo só para gravar um shard minúsculo
        if prog["tokens"] >= plan.quota or prog["exhausted"] or prog["complete"]:
            break
        if file in prog["files_done"]:
            continue
        shard = src_dir / f"{plan.name}_{i:05d}.bin"
        tmp = shard.with_suffix(".bin.tmp")
        ids_file = shard.with_suffix(".ids")
        t0 = time.time()
        n_tok = n_docs = n_skip = 0
        room = plan.quota - prog["tokens"]
        kept_ids: list[str] = []
        full = False
        with open(tmp, "wb") as f:
            for batch in _prefetch(_batches(read_docs(file), _ENCODE_DOCS), _PREFETCH):
                if skip_ids:
                    before = len(batch)
                    batch = [d for d in batch if d[0] not in skip_ids]
                    n_skip += before - len(batch)
                encs = tokenizer.encode_batch([t for _, t in batch], add_special_tokens=False)
                chunks = []
                for (doc_id, _), enc in zip(batch, encs):
                    if not enc.ids:
                        continue
                    if n_tok + len(enc.ids) + 1 > room:
                        full = True
                        break
                    arr = np.empty(len(enc.ids) + 1, dtype=np.int64)
                    arr[:-1] = enc.ids
                    arr[-1] = eos_id
                    chunks.append(arr)
                    n_tok += len(arr)
                    n_docs += 1
                    if save_ids and doc_id is not None:
                        kept_ids.append(doc_id)
                if chunks:
                    out = np.concatenate(chunks)
                    if out.max() > limit:
                        raise ValueError(f"id {out.max()} não cabe em {dtype}")
                    f.write(out.astype(np_dtype).tobytes())
                if full:
                    break
            f.flush()
            os.fsync(f.fileno())
        if n_tok == 0:
            tmp.unlink()
        else:
            if save_ids:
                ids_file.write_text("\n".join(kept_ids))
            shard.with_suffix(".json").write_text(json.dumps({"n_tokens": n_tok, "dtype": dtype}))
            os.replace(tmp, shard)
        prog["tokens"] += n_tok
        prog["files_done"].append(file)
        prog["complete"] = full
        _save_json(prog_path, prog)
        dt = time.time() - t0
        log(
            f"[{plan.name}] arquivo {i + 1}/{len(plan.files)}: {n_tok:,} tokens, {n_docs:,} docs"
            + (f", {n_skip:,} duplicados pulados" if skip_ids else "")
            + f" ({n_tok / max(dt, 1e-9):,.0f} tok/s) | fonte {prog['tokens']:,}/{plan.quota:,}"
        )
        if full:
            break
    else:
        if prog["tokens"] < plan.quota and not prog["exhausted"] and not prog["complete"]:
            prog["exhausted"] = True
            _save_json(prog_path, prog)
            log(f"[{plan.name}] AVISO: fonte esgotada com {prog['tokens']:,} de {plan.quota:,} tokens")
    return prog["tokens"]


def load_ids(src_dir: Path) -> set[str]:
    """Ids gravados pelos shards prontos de uma fonte (`*.ids`)."""
    ids: set[str] = set()
    for p in sorted(src_dir.glob("*.ids")):
        if p.with_suffix(".bin").exists():
            ids.update(line for line in p.read_text().splitlines() if line)
    return ids
