#!/usr/bin/env python3
"""
soreRNN-LM: Pipeline de SFT Multi-Tarefa com Máscara de Loss e Validação Held-Out.
Combina:
- Conversação Geral em Português (Alpaca-PT / Dolly-PT)
- Diálogos de Matemática com Raciocínio CoT e Alta Variabilidade Sintática
- Problemas de Lógica e Dedução
"""

import os
import random
import argparse
import numpy as np
from datasets import load_dataset

from tokenizer_utils import get_tokenizer
from dedup_filter import MinHashDeduplicator
from math_dataset_utils import generate_diverse_synthetic_math

def generate_logic_sft_pairs(count: int = 6000):
    pairs = []
    itens = ["maçãs", "laranjas", "livros", "canetas", "moedas", "cadernos", "lápis"]
    nomes = [("Lucas", "seu irmão"), ("Maria", "sua amiga"), ("Pedro", "seu colega"), ("Ana", "sua prima")]

    for _ in range(count):
        item = random.choice(itens)
        n1 = random.randint(15, 90)
        n2 = random.randint(3, n1 - 2)
        sobra = n1 - n2
        nome, amigo = random.choice(nomes)

        q = f"{nome} tinha {n1} {item} e deu {n2} para {amigo}. Com quantas {item} {nome} ficou?"
        r = (
            f"Vamos resolver passo a passo:\n"
            f"1. Quantidade inicial: {n1} {item}.\n"
            f"2. Quantidade doada: {n2} {item}.\n"
            f"3. Subtração: {n1} - {n2} = {sobra}.\n"
            f"Resposta: {nome} ficou com {sobra} {item}."
        )
        pairs.append((f"<|user|>\n{q}\n<|assistant|>\n", f"{r}\n<|endoftext|>\n"))
    return pairs

def build_multitask_sft(
    out_path: str = "data/sft_multitask_test.bin",
    val_out_path: str = "data/sft_multitask_val.bin",
    target_tokens: int = 1_500_000,
    val_tokens: int = 100_000,
    tokenizer_name: str = "pt"
):
    tok = get_tokenizer(tokenizer_name)
    dedup = MinHashDeduplicator()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if val_out_path:
        os.makedirs(os.path.dirname(os.path.abspath(val_out_path)), exist_ok=True)

    print("=" * 75)
    print("  GERANDO DATASET SFT MULTI-TAREFA COM MÁSCARA E VALIDAÇÃO HELD-OUT")
    print("=" * 75)

    chat_pairs = []
    try:
        ds = load_dataset("dominguesm/alpaca-data-pt-br", split="train")
        for row in ds:
            inst = str(row.get("instruction") or "").strip()
            inp = str(row.get("input") or "").strip()
            out = str(row.get("output") or "").strip()
            if not inst or not out:
                continue
            user_msg = f"{inst} {inp}".strip()
            chat_pairs.append((f"<|user|>\n{user_msg}\n<|assistant|>\n", f"{out}\n<|endoftext|>\n"))
            if len(chat_pairs) >= 30000:
                break
    except Exception as e:
        print(f" -> Aviso ao carregar Alpaca-PT: {e}")

    math_pairs = []
    for _ in range(12000):
        m = generate_diverse_synthetic_math()
        pts = m.split("\n\n", 1)
        q = pts[0] if len(pts) == 2 else m
        a = pts[1] if len(pts) == 2 else m
        math_pairs.append((f"<|user|>\n{q}\n<|assistant|>\n", f"{a}\n<|endoftext|>\n"))

    logic_pairs = generate_logic_sft_pairs(count=6000)

    all_pairs = chat_pairs + math_pairs + logic_pairs
    random.seed(42)
    random.shuffle(all_pairs)

    def write_dataset(pairs_slice, fpath, max_toks, is_val=False):
        written = 0
        buf = []
        p_idx = 0
        with open(fpath, "wb") as f:
            while written < max_toks:
                if p_idx >= len(pairs_slice):
                    random.shuffle(pairs_slice)
                    p_idx = 0
                prompt, resp = pairs_slice[p_idx]
                p_idx += 1

                full_t = prompt + resp
                if is_val:
                    if not dedup.add_validation_document(full_t):
                        continue
                else:
                    if dedup.is_val_leak(full_t):
                        continue

                p_toks = tok.encode(prompt, allowed_special={"<|endoftext|>"})
                r_toks = tok.encode(resp, allowed_special={"<|endoftext|>"})
                seq = p_toks + r_toks

                for i in range(len(seq) - 1):
                    inp = seq[i]
                    tgt = 65535 if i < len(p_toks) - 1 else seq[i + 1]
                    buf.append(inp)
                    buf.append(tgt)
                    written += 1
                    if written >= max_toks:
                        break

            if buf:
                np.array(buf, dtype=np.uint16).tofile(f)

    split_n = min(len(all_pairs) // 10, 2000)
    val_slice = all_pairs[:split_n]
    train_slice = all_pairs[split_n:]

    if val_out_path and val_tokens > 0:
        write_dataset(val_slice, val_out_path, val_tokens, is_val=True)
        print(f" -> Validação SFT Multi-tarefa: {val_out_path}")

    write_dataset(train_slice, out_path, target_tokens, is_val=False)
    print(f" -> Treino SFT Multi-tarefa: {out_path} ({target_tokens:,} tokens)")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera dataset SFT multi-tarefa")
    parser.add_argument("--out", type=str, default="data/sft_multitask_test.bin")
    parser.add_argument("--val_out", type=str, default="data/sft_multitask_val.bin")
    parser.add_argument("--tokens", type=int, default=1_500_000)
    parser.add_argument("--val_tokens", type=int, default=100_000)
    parser.add_argument("--tokenizer", type=str, default="pt", choices=["pt", "gpt2", "llama"])
    args = parser.parse_args()

    build_multitask_sft(
        out_path=args.out,
        val_out_path=args.val_out,
        target_tokens=args.tokens,
        val_tokens=args.val_tokens,
        tokenizer_name=args.tokenizer
    )
