"""Gera a amostra real de texto PT (tokenizer Qwen3.5, vocab remapeado para 65.536) para os portões de qualidade.

Fonte: mC4 português (`allenai/c4`, `multilingual/c4-pt.tfrecord-XXXXX-of-01024.json.gz`, ODC-BY),
lido em streaming com gzip+json da stdlib; baixa só os arquivos necessários para chegar a `--n-tokens`.
Saída em `--out`: `sample_000.bin`(+`.json`), `remap.npy`, `doc_lengths.json` e `meta.json`.
Os brutos ficam em `data/raw/` (dados não confiáveis: só lidos como gzip/JSON).

Uso: python scripts/make_text_sample.py --out data/sample_pt --n-tokens 3e8 \
         [--tokenizer Qwen/Qwen3.5-0.8B] [--revision main] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tokenizers import Tokenizer

from sore.train.text_sample import download, generate_sample, iter_remote_texts, resolve_revision

DATASET = "allenai/c4"
N_FILES = 1024
URL = "https://huggingface.co/datasets/{repo}/resolve/{rev}/multilingual/c4-pt.tfrecord-{i:05d}-of-{n:05d}.json.gz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-tokens", type=float, default=3e8)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    n_tokens = int(args.n_tokens)
    if n_tokens < 1:
        parser.error("--n-tokens deve ser >= 1")

    revision = resolve_revision(args.tokenizer, args.revision)
    tok_path = download(
        f"https://huggingface.co/{args.tokenizer}/resolve/{revision}/tokenizer.json",
        args.raw_dir / f"tokenizer-{args.tokenizer.replace('/', '--')}-{revision[:12]}.json",
    )
    tokenizer = Tokenizer.from_file(str(tok_path))
    eos_old = tokenizer.token_to_id("<|endoftext|>")
    if eos_old is None:
        parser.error(f"{args.tokenizer}: token '<|endoftext|>' ausente no tokenizer")

    dataset_rev = resolve_revision(DATASET, "main", kind="datasets")
    urls = [URL.format(repo=DATASET, rev=dataset_rev, i=i, n=N_FILES) for i in range(N_FILES)]
    used: list[str] = []
    t0 = time.time()
    stats = generate_sample(
        iter_remote_texts(urls, args.raw_dir, used.append),
        tokenizer,
        eos_old,
        args.out,
        n_tokens,
        seed=args.seed,
    )
    meta = {
        "source_urls": used,
        "tokenizer_repo": args.tokenizer,
        "tokenizer_revision": revision,
        "eos_old_id": eos_old,
        **stats,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"{args.out}: {stats['n_tokens']} tokens, {stats['n_docs']} docs, unk_rate={stats['unk_rate']:.5f} "
          f"({time.time() - t0:.0f}s, {len(used)} arquivo(s))")


if __name__ == "__main__":
    main()
