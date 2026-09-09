#!/usr/bin/env python3
"""
soreRNN-LM: Pipeline de SFT em Larga Escala (500M Tokens) com Máscara de Loss Cirúrgica
e Conjunto de Validação Held-Out para Prevenção Rigorosa de Memorização.

Gera pares uint16_t [input, target] para DataLoader(is_paired=True):
  - Tokens do prompt recebem target = 65535 (IGNORE_INDEX no kernel CUDA)
  - Tokens da resposta recebem o próximo token real
  - Separação estrita de validação (data/sft_val.bin) com deduplicação
"""

import os
import sys
import time
import random
import argparse
import numpy as np
from datasets import load_dataset

from tokenizer_utils import get_tokenizer
from dedup_filter import MinHashDeduplicator
from math_dataset_utils import generate_diverse_synthetic_math

def load_conversational_pairs():
    """Carrega dados conversacionais em português (Alpaca-PT e Dolly-PT)."""
    pairs = []
    print(" -> Carregando diálogos instrucionais em português...")
    for repo, split in [("dominguesm/alpaca-data-pt-br", "train"), ("databricks/databricks-dolly-15k", "train")]:
        try:
            ds = load_dataset(repo, split=split)
            for row in ds:
                inst = str(row.get("instruction") or row.get("pergunta") or row.get("prompt") or "").strip()
                inp = str(row.get("input") or row.get("contexto") or "").strip()
                out = str(row.get("output") or row.get("resposta") or row.get("response") or "").strip()
                if not inst or not out:
                    continue
                user_txt = f"{inst}\n{inp}".strip() if inp else inst
                prompt = f"<|user|>\n{user_txt}\n<|assistant|>\n"
                resp = f"{out}\n<|endoftext|>\n"
                pairs.append((prompt, resp))
        except Exception as e:
            print(f" -> [Aviso] Falha ao carregar {repo}: {e}")

    print(f" -> Total de diálogos conversacionais carregados: {len(pairs):,}")
    return pairs

def generate_math_sft_pairs(count: int = 50_000):
    """Gera diálogos matemáticos com raciocínio passo a passo diversificado."""
    pairs = []
    print(f" -> Gerando {count:,} diálogos matemáticos de alta entropia...")
    for _ in range(count):
        math_text = generate_diverse_synthetic_math()
        # Divide na primeira quebra dupla ou transição
        parts = math_text.split("\n\n", 1)
        if len(parts) == 2:
            q, a = parts[0], parts[1]
        else:
            q, a = math_text, math_text
        prompt = f"<|user|>\n{q}\n<|assistant|>\n"
        resp = f"{a}\n<|endoftext|>\n"
        pairs.append((prompt, resp))
    return pairs

def build_500m_sft_dataset(
    out_path: str = "data/sft_500m.bin",
    val_out_path: str = "data/sft_val.bin",
    target_tokens: int = 500_000_000,
    val_tokens: int = 500_000,
    tokenizer_name: str = "pt",
    chunk_tokens: int = 250_000
):
    tok = get_tokenizer(tokenizer_name)
    eos_id = tok.eos_token_id
    dedup = MinHashDeduplicator()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if val_out_path:
        os.makedirs(os.path.dirname(os.path.abspath(val_out_path)), exist_ok=True)

    print("=" * 80)
    print("      soreRNN-LM: GERADOR DE SFT COM LOSS MASKING E VALIDAÇÃO HELD-OUT    ")
    print("=" * 80)
    print(f" Tokenizer:         {tokenizer_name} (vocab={tok.vocab_size}, eos_id={eos_id})")
    print(f" Arquivo Treino:    {out_path} ({target_tokens:,} tokens)")
    print(f" Arquivo Validação: {val_out_path} ({val_tokens:,} tokens)")
    print("=" * 80)

    conv_pairs = load_conversational_pairs()
    math_pairs = generate_math_sft_pairs(count=60_000)
    pool = conv_pairs + math_pairs

    if not pool:
        print("[Erro] Nenhum par de diálogo disponível para SFT.", file=sys.stderr)
        return

    random.seed(42)
    random.shuffle(pool)

    # 1. Conjunto de Validação Held-Out (Deduplicado)
    if val_out_path and val_tokens > 0 and not os.path.exists(val_out_path):
        print(f"\n[1/2] Gravando conjunto de validação ({val_out_path})...")
        val_written = 0
        val_buf = []
        with open(val_out_path, "wb") as f_val:
            for prompt, resp in pool:
                if val_written >= val_tokens:
                    break
                full_text = prompt + resp
                if not dedup.add_validation_document(full_text):
                    continue

                p_toks = tok.encode(prompt, allowed_special={"<|endoftext|>"})
                r_toks = tok.encode(resp, allowed_special={"<|endoftext|>"})
                seq = p_toks + r_toks

                for i in range(len(seq) - 1):
                    inp = seq[i]
                    tgt = 65535 if i < len(p_toks) - 1 else seq[i + 1]
                    val_buf.append(inp)
                    val_buf.append(tgt)
                    val_written += 1

            if val_buf:
                np.array(val_buf, dtype=np.uint16).tofile(f_val)
        print(f" -> Validação SFT concluída: {val_written:,} tokens mascarados gravados.")

    # 2. Conjunto de Treinamento
    bytes_per_pair = 4
    existing_pairs = os.path.getsize(out_path) // bytes_per_pair if os.path.exists(out_path) else 0

    if existing_pairs >= target_tokens:
        print(f"Meta de {target_tokens:,} tokens SFT já atingida em {out_path}.")
        return

    print(f"\n[2/2] Gravando treino SFT com Loss Masking ({target_tokens - existing_pairs:,} restantes)...")
    total_written = existing_pairs
    mode = "ab" if existing_pairs > 0 else "wb"

    pool_idx = 0
    t_start = time.time()
    t_last = time.time()
    tokens_since_last = 0

    with open(out_path, mode) as f_out:
        while total_written < target_tokens:
            chunk_buf = []
            while len(chunk_buf) < (chunk_tokens * 2) and total_written + len(chunk_buf) // 2 < target_tokens:
                if pool_idx >= len(pool):
                    random.shuffle(pool)
                    pool_idx = 0

                prompt, resp = pool[pool_idx]
                pool_idx += 1

                full_text = prompt + resp
                # Evita vazamento para a validação
                if dedup.is_val_leak(full_text):
                    continue

                p_toks = tok.encode(prompt, allowed_special={"<|endoftext|>"})
                r_toks = tok.encode(resp, allowed_special={"<|endoftext|>"})
                seq = p_toks + r_toks

                for i in range(len(seq) - 1):
                    inp = seq[i]
                    tgt = 65535 if i < len(p_toks) - 1 else seq[i + 1]
                    chunk_buf.append(inp)
                    chunk_buf.append(tgt)

            arr = np.array(chunk_buf, dtype=np.uint16)
            arr.tofile(f_out)
            f_out.flush()

            added = len(arr) // 2
            total_written += added
            tokens_since_last += added

            now = time.time()
            elapsed = max(now - t_last, 0.001)
            if elapsed >= 5.0 or total_written >= target_tokens:
                speed = tokens_since_last / elapsed
                speed_global = (total_written - existing_pairs) / max(now - t_start, 0.001)
                pct = (total_written / target_tokens) * 100.0
                rem = target_tokens - total_written
                eta_h = (rem / max(speed_global, 1.0)) / 3600.0

                print(f"[{pct:5.2f}%] SFT Tokens: {total_written:,}/{target_tokens:,} | Vel: {speed:,.0f} tok/s | ETA: {eta_h:.1f}h")
                t_last = now
                tokens_since_last = 0

    print("\n" + "=" * 80)
    print(f" [SUCESSO] Dataset SFT Finalizado: {out_path} ({total_written:,} tokens)")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera dataset SFT com loss masking e validação")
    parser.add_argument("--out", type=str, default="data/sft_500m.bin")
    parser.add_argument("--val_out", type=str, default="data/sft_val.bin")
    parser.add_argument("--tokens", type=int, default=500_000_000)
    parser.add_argument("--val_tokens", type=int, default=500_000)
    parser.add_argument("--tokenizer", type=str, default="pt", choices=["pt", "gpt2", "llama"])
    parser.add_argument("--chunk_size", type=int, default=250_000)
    args = parser.parse_args()

    build_500m_sft_dataset(
        out_path=args.out,
        val_out_path=args.val_out,
        target_tokens=args.tokens,
        val_tokens=args.val_tokens,
        tokenizer_name=args.tokenizer,
        chunk_tokens=args.chunk_size
    )
