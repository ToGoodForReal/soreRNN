"""Gera os shards de pré-treino (`--total-tokens`, mistura de `MIXTURE`) com o tokenizer podado.

Lê `--tokenizer-dir` (`tokenizer.json` e `meta.json` do `scripts/prune_tokenizer.py`), resolve as
revisões dos datasets (fixadas em `<out>/manifest.json` na 1ª execução, para a retomada ler os
mesmos arquivos) e grava `<out>/<fonte>/<fonte>_NNNNN.bin` (+ `.json`). O dtype dos shards é
`uint16` se o vocabulário couber em 65.536 ids, senão `uint32`. Recusa começar sem espaço livre
para o que falta gerar. Pode ser interrompido e rodado de novo: continua de onde parou.

Uso: uv run python scripts/make_pretrain_data.py --tokenizer-dir data/tokenizer \
         --out /caminho/pretrain [--total-tokens 50e9] [--only fonte1,fonte2]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi
from tokenizers import Tokenizer

from sore.train.pretrain_data import DEDUP_AGAINST, SourcePlan, load_ids, plan_order, write_source
from sore.train.sources import MIXTURE, SOURCES, hub_path, iter_documents, list_files
from sore.train.tokenizer_prune import to_json_str, without_added_tokens

DISK_MARGIN = 1.05


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--total-tokens", type=float, default=50e9)
    parser.add_argument("--only", default="", help="gera só estas fontes (separadas por vírgula)")
    args = parser.parse_args()

    tok_meta = json.loads((args.tokenizer_dir / "meta.json").read_text())
    tok_json = json.loads((args.tokenizer_dir / "tokenizer.json").read_text())
    tokenizer = Tokenizer.from_str(to_json_str(without_added_tokens(tok_json)))  # "<|im_start|>" literal vira texto
    eos_id, vocab = tok_meta["eos_id"], tok_meta["vocab_size"]
    dtype = "uint16" if vocab <= 1 << 16 else "uint32"
    width = 2 if dtype == "uint16" else 4
    total = int(args.total_tokens)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["total_tokens"], manifest["eos_id"], manifest["dtype"]) != (total, eos_id, dtype):
            parser.error(f"{manifest_path} foi gerado com outro total/eos/dtype; use outro --out")
    else:
        api = HfApi()
        revisions = {n: api.dataset_info(SOURCES[n].repo).sha for n in MIXTURE}
        manifest = {
            "total_tokens": total, "eos_id": eos_id, "vocab_size": vocab, "dtype": dtype,
            "tokenizer_dir": str(args.tokenizer_dir.resolve()), "tokenizer_meta": tok_meta,
            "mixture": MIXTURE, "revisions": revisions,
            "files": {n: list_files(SOURCES[n], revisions[n], api) for n in MIXTURE},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    names = plan_order([n for n in MIXTURE if not args.only or n in args.only.split(",")])
    done = {n: _done_tokens(args.out / n) for n in MIXTURE}
    missing = sum(max(0, int(MIXTURE[n] * total) - done[n]) for n in names) * width
    free = shutil.disk_usage(args.out).free
    print(f"[pretrain] {dtype}, eos_id {eos_id} | faltam ~{missing / 1e9:.1f} GB, livres {free / 1e9:.1f} GB", flush=True)
    if missing * DISK_MARGIN > free:
        parser.error(f"espaço insuficiente em {args.out}: precisa de ~{missing * DISK_MARGIN / 1e9:.0f} GB")

    t0 = time.time()
    for name in names:
        src, rev = SOURCES[name], manifest["revisions"][name]
        plan = SourcePlan(name, int(MIXTURE[name] * total), manifest["files"][name])
        skip = load_ids(args.out / DEDUP_AGAINST[name]) if name in DEDUP_AGAINST else None
        if skip is not None:
            print(f"[{name}] dedup contra {DEDUP_AGAINST[name]}: {len(skip):,} ids", flush=True)
        got = write_source(
            plan, args.out, tokenizer, eos_id, dtype,
            read_docs=lambda path, src=src, rev=rev: iter_documents(src, hub_path(src, rev, path)),
            skip_ids=skip, save_ids=name in DEDUP_AGAINST.values(),
            log=lambda msg: print(msg, flush=True),
        )
        print(f"[{name}] pronto: {got:,} tokens ({(time.time() - t0) / 3600:.1f} h desde o início)", flush=True)
    totals = {n: _done_tokens(args.out / n) for n in MIXTURE}
    print(f"[pretrain] total {sum(totals.values()):,} tokens: " + ", ".join(f"{n} {v / 1e9:.2f}B" for n, v in totals.items()), flush=True)


def _done_tokens(src_dir: Path) -> int:
    p = src_dir / "progress.json"
    return json.loads(p.read_text())["tokens"] if p.exists() else 0


if __name__ == "__main__":
    main()
