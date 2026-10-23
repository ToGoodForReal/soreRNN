"""Gera um shard sintético uint32 para smoke-tests locais do pipeline de dados e treino.

Documentos de comprimento aleatório (8-600 tokens de conteúdo) com padrão aprendível:
progressão aritmética módulo o vocabulário (sem nunca emitir o EOS), cada documento
terminado por EOS. O shard tem exatamente `--n-tokens` tokens e termina em EOS.

Uso: python scripts/make_synthetic_shard.py --out data/synth.bin --n-tokens 2000000 \
         --vocab 512 --eos 0 --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sore.train.data import write_token_shard

MIN_DOC, MAX_DOC = 8, 600


def make_tokens(n_tokens: int, vocab: int, eos: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_content = vocab - 1  # ids de conteúdo: todos menos o EOS
    out = np.empty(n_tokens, dtype=np.uint32)
    pos = 0
    while pos < n_tokens:
        # reserva 1 posição para o EOS; o último documento é truncado para fechar em n_tokens
        length = min(int(rng.integers(MIN_DOC, MAX_DOC + 1)), n_tokens - pos - 1)
        start, step = int(rng.integers(0, n_content)), int(rng.integers(1, 8))
        raw = (start + step * np.arange(length)) % n_content
        out[pos : pos + length] = raw + (raw >= eos)  # pula o id do EOS
        out[pos + length] = eos
        pos += length + 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-tokens", type=int, default=2_000_000)
    parser.add_argument("--vocab", type=int, default=512)
    parser.add_argument("--eos", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.n_tokens < 1:
        parser.error("--n-tokens deve ser >= 1")
    if args.vocab < 3 or not 0 <= args.eos < args.vocab:
        parser.error("exige --vocab >= 3 e 0 <= --eos < --vocab")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_token_shard(args.out, make_tokens(args.n_tokens, args.vocab, args.eos, args.seed))
    print(f"{args.out}: {args.n_tokens} tokens (vocab={args.vocab}, eos={args.eos})")


if __name__ == "__main__":
    main()
