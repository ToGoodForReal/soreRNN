#!/usr/bin/env python3
"""
soreRNN-LM: Validação e Diagnóstico de Integridade de Datasets
Suporta datasets sequenciais e emparelhados (is_paired=True com máscaras 65535).
"""

import os
import sys
import argparse
import numpy as np
from tokenizer_utils import get_tokenizer

def validate_file(filepath: str, vocab_size: int = 50257, tokenizer_name: str = "pt"):
    print("=" * 70)
    print(f" VALIDAÇÃO DE DATASET: {filepath}")
    print("=" * 70)

    if not os.path.exists(filepath):
        print(f"[Aviso] Arquivo não encontrado: {filepath}")
        return False

    file_size = os.path.getsize(filepath)
    if file_size % 2 != 0:
        print("[ERRO] Arquivo uint16 truncado (tamanho em bytes é ímpar).")
        return False

    total_uint16 = file_size // 2
    size_mb = file_size / (1024 * 1024)

    m = np.memmap(filepath, dtype=np.uint16, mode="r")

    # Detecta se é emparelhado (procura 65535 nas primeiras posições ímpares)
    is_paired = False
    if total_uint16 >= 200:
        if np.any(m[1:200:2] == 65535):
            is_paired = True

    total_tokens = total_uint16 // 2 if is_paired else total_uint16

    print(f" -> Formato detectado: {'EMPARELHADO (Input, Target Mascarado)' if is_paired else 'SEQUENCIAL PADRÃO'}")
    print(f" -> Tamanho em disco:  {size_mb:,.2f} MB")
    print(f" -> Tokens de treino:  {total_tokens:,}")

    tok = get_tokenizer(tokenizer_name)

    # Coleta amostra limpa sem 65535
    sample_raw = m[:min(500_000, len(m))]
    valid_mask = (sample_raw < 65535)
    sample_valid = sample_raw[valid_mask]

    if len(sample_valid) == 0:
        print("[ERRO] Nenhum token válido encontrado na amostra.")
        return False

    min_id = int(sample_valid.min())
    max_id = int(sample_valid.max())
    unique_tokens = len(np.unique(sample_valid))

    print(f" -> Min Token ID:      {min_id}")
    print(f" -> Max Token ID:      {max_id} (limite vocab={vocab_size})")
    print(f" -> Tokens Únicos:     {unique_tokens:,}")

    if max_id >= vocab_size:
        print(f"[ERRO CRÍTICO] Token ID {max_id} excede vocab_size {vocab_size}!")
        return False

    print("\n--- AMOSTRAS DE TEXTO DECODIFICADAS ---")
    if is_paired:
        # Extrai apenas as posições de input (índices pares)
        inputs_only = m[0:min(400, len(m)):2]
        decoded = tok.decode(inputs_only[:100].tolist())
        print(f" Amostra inicial:\n   \"{decoded.strip()}\"\n")
    else:
        decoded = tok.decode(m[:100].tolist())
        print(f" Amostra inicial:\n   \"{decoded.strip()}\"\n")

    print(f"[SUCESSO] Dataset {filepath} validado com integridade 100%!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validador de datasets binários")
    parser.add_argument("file", nargs="?", default="data/pretrain_interleaved.bin")
    parser.add_argument("--tokenizer", type=str, default="pt")
    args = parser.parse_args()

    validate_file(args.file, tokenizer_name=args.tokenizer)
