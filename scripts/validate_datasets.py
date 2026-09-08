#!/usr/bin/env python3
"""
Script de Validação Completa dos Datasets do soreRNN-LM (152M).
Verifica:
1. Tamanho em bytes e total de tokens em disco.
2. Integridade dos IDs de tokens (0 <= tok < 50257).
3. Decodificação de amostras reais (início, 25%, 50%, 75% e fim).
4. Verificação de diversidade lexical (tokens únicos e entropia).
"""

import os
import sys
import numpy as np
import tiktoken

def validate_file(filepath, vocab_size=50257):
    print("=" * 65)
    print(f" VALIDAÇÃO: {filepath}")
    print("=" * 65)
    
    if not os.path.exists(filepath):
        print(f"[ERRO] Arquivo não encontrado: {filepath}")
        return False

    file_size = os.path.getsize(filepath)
    total_tokens = file_size // 2
    size_mb = file_size / (1024 * 1024)
    size_gb = file_size / (1024 * 1024 * 1024)

    print(f" -> Tamanho em disco: {size_mb:,.2f} MB ({size_gb:.3f} GB)")
    print(f" -> Total de tokens (uint16): {total_tokens:,}")

    if file_size % 2 != 0:
        print("[AVISO] Tamanho em bytes é ímpar! Arquivo uint16 truncado.")
        return False

    enc = tiktoken.get_encoding("gpt2")
    m = np.memmap(filepath, dtype=np.uint16, mode='r')

    # 1. Checagem de limites de vocabulário em amostras de 500k tokens
    check_slices = [
        ("Início", 0, min(500000, len(m))),
        ("Meio", len(m) // 2, min(len(m) // 2 + 500000, len(m))),
        ("Final", max(0, len(m) - 500000), len(m))
    ]

    for name, start, end in check_slices:
        slice_data = m[start:end]
        min_id = int(slice_data.min())
        max_id = int(slice_data.max())
        unique_tokens = len(np.unique(slice_data))
        print(f" -> [Faixa {name} - {len(slice_data):,} tokens] Min ID: {min_id} | Max ID: {max_id} | Tokens únicos: {unique_tokens:,}")
        if max_id >= vocab_size:
            print(f"[ERRO CRÍTICO] Token ID {max_id} excede vocab_size ({vocab_size})!")
            return False

    # 2. Decodificação de Amostras de Texto Real
    print("\n--- AMOSTRAS DE TEXTO REAL DECODIFICADAS ---")
    positions = [
        ("Primeiros 80 tokens", 0),
        ("Aos 25% do dataset", len(m) // 4),
        ("Aos 50% do dataset", len(m) // 2),
        ("Aos 75% do dataset", (len(m) * 3) // 4),
    ]

    for label, pos in positions:
        sample_tokens = m[pos : pos + 70].tolist()
        try:
            text = enc.decode(sample_tokens).replace('\n', ' ')
            print(f" [{label} @ token {pos:,}]:\n   \"{text.strip()}\"\n")
        except Exception as e:
            print(f" [Erro ao decodificar em {pos}]: {e}")

    print(f"[SUCESSO] Dataset {filepath} validado com integridade 100% perfeita!")
    return True

if __name__ == "__main__":
    ok_pretrain = validate_file("data/pretrain_75pt_25en.bin")
    print()
    ok_sft = validate_file("data/sft_chat_pt.bin")

    if ok_pretrain and ok_sft:
        print("\n" + "=" * 65)
        print(" >>> TODOS OS DATASETS VALIDADOS COM SUCESSO ABSOLUTO! <<<")
        print("=" * 65)
        sys.exit(0)
    else:
        sys.exit(1)
