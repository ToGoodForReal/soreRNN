#!/usr/bin/env python3
"""
soreRNN-LM: Pré-Treino com Interleaving Dinâmico de Grão Fino e Validação Held-Out
Gera dados com multiplexação contínua documento-a-documento:
  - 80% Português Enriquecido (Wikipedia, C4-PT, Madras1)
  - 10% Ciência e Educação (FineWeb-Edu com score >= 3.0)
  - 10% Matemática Real e Raciocínio Procedural de Alta Entropia
"""

import os
import sys
import argparse
from prepare_4b_pretrain import generate_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera dataset de pré-treino com interleaving de grão fino por documento")
    parser.add_argument("--out", type=str, default="data/pretrain_interleaved.bin")
    parser.add_argument("--val_out", type=str, default="data/val.bin")
    parser.add_argument("--tokens", type=int, default=50_000_000, help="Total de tokens alvo de treino")
    parser.add_argument("--val_tokens", type=int, default=500_000, help="Total de tokens de validação held-out")
    parser.add_argument("--tokenizer", type=str, default="pt", choices=["pt", "gpt2", "llama"])
    parser.add_argument("--pt_weight", type=float, default=0.80)
    parser.add_argument("--edu_weight", type=float, default=0.10)
    parser.add_argument("--math_weight", type=float, default=0.10)
    parser.add_argument("--paired", action="store_true", default=True)
    parser.add_argument("--no_paired", dest="paired", action="store_false")
    args = parser.parse_args()

    generate_dataset(
        out_path=args.out,
        val_out_path=args.val_out,
        target_tokens=args.tokens,
        val_tokens=args.val_tokens,
        tokenizer_name=args.tokenizer,
        pt_weight=args.pt_weight,
        edu_weight=args.edu_weight,
        math_weight=args.math_weight,
        paired_mode=args.paired
    )
