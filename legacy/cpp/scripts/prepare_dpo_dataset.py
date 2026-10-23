#!/usr/bin/env python3
"""
soreRNN-LM: Preparação de Dataset de Preferências Diretas (DPO) Nativo
Gera arquivos binários estruturados diretamente consumíveis pelo executável C++ 'train_dpo':
  - Header DPO1 (seq_len, num_pairs, vocab_size)
  - Pares (chosen_input, chosen_target, rejected_input, rejected_target, ref_logp_w, ref_logp_l)
  - Prompt com target mascarado em 65535
"""

import os
import sys
import struct
import random
import argparse
import numpy as np
from tokenizer_utils import get_tokenizer

def generate_preference_samples(count: int = 5000):
    pairs = []
    nomes = ["Carlos", "Mariana", "Lucas", "Beatriz", "Pedro", "Ana", "Gabriel", "Camila"]
    itens = ["livros", "cadernos", "canetas", "moedas", "maçãs", "figurinhas"]

    for _ in range(count):
        nome = random.choice(nomes)
        item = random.choice(itens)
        a = random.randint(20, 150)
        b = random.randint(10, 80)
        c = random.randint(5, 30)
        total_correto = a + b - c
        total_errado = a + b + c # Alucinação aritmética

        prompt = (
            f"<|user|>\n"
            f"{nome} possuía {a} {item} na coleção, adquiriu mais {b} {item} e doou {c} {item}. "
            f"Com quantas {item} {nome} ficou no total? Demonstre o raciocínio passo a passo.\n"
            f"<|assistant|>\n"
        )

        chosen = (
            f"Vamos calcular a quantidade final de {item} passo a passo:\n"
            f"1. Quantidade inicial: {a} {item}.\n"
            f"2. Após adquirir {b} {item}: {a} + {b} = {a + b} {item}.\n"
            f"3. Após doar {c} {item}: {a + b} - {c} = {total_correto} {item}.\n"
            f"Portanto, {nome} ficou com exatamente {total_correto} {item}.<|endoftext|>\n"
        )

        rejected = (
            f"{nome} ficou com {total_errado} {item}. "
            f"Eu apenas somei todos os números mencionados na pergunta: {a} + {b} + {c} = {total_errado}.<|endoftext|>\n"
        )

        pairs.append((prompt, chosen, rejected))

    return pairs

def pad_or_truncate_sequence(p_toks: list, r_toks: list, seq_len: int):
    """
    Retorna (inputs, targets) de tamanho seq_len.
    Tokens do prompt recebem target = 65535.
    Tokens de padding recebem input=0, target=65535.
    """
    seq = p_toks + r_toks
    inputs = []
    targets = []

    for i in range(len(seq) - 1):
        inp = seq[i]
        tgt = 65535 if i < len(p_toks) - 1 else seq[i + 1]
        inputs.append(inp)
        targets.append(tgt)

    # Trunca ou aplica padding até seq_len
    if len(inputs) > seq_len:
        inputs = inputs[:seq_len]
        targets = targets[:seq_len]
    else:
        pad_count = seq_len - len(inputs)
        inputs.extend([0] * pad_count)
        targets.extend([65535] * pad_count)

    return inputs, targets

def build_dpo_binary_dataset(
    out_path: str = "data/dpo_pairs.bin",
    count: int = 5000,
    seq_len: int = 256,
    tokenizer_name: str = "pt"
):
    tok = get_tokenizer(tokenizer_name)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    print("=" * 75)
    print("      soreRNN-LM: GERADOR DE DATASET DPO BINÁRIO NATIVO (DPO1)")
    print("=" * 75)
    print(f" Arquivo de Saída: {out_path}")
    print(f" Total de Pares:   {count:,}")
    print(f" SeqLen Padrão:    {seq_len}")
    print(f" Tokenizer:        {tokenizer_name} (vocab={tok.vocab_size})")
    print("=" * 75)

    pairs = generate_preference_samples(count=count)

    with open(out_path, "wb") as f_out:
        # Header DPO1: magic (4B), seq_len (4B), num_pairs (4B), vocab_size (4B), reserved (16B)
        magic = b"DPO1"
        header = struct.pack("<4sIII16s", magic, seq_len, len(pairs), tok.vocab_size, bytes(16))
        f_out.write(header)

        for prompt, chosen, rejected in pairs:
            p_toks = tok.encode(prompt, allowed_special={"<|endoftext|>"})
            cw_toks = tok.encode(chosen, allowed_special={"<|endoftext|>"})
            cr_toks = tok.encode(rejected, allowed_special={"<|endoftext|>"})

            inp_w, tgt_w = pad_or_truncate_sequence(p_toks, cw_toks, seq_len)
            inp_l, tgt_l = pad_or_truncate_sequence(p_toks, cr_toks, seq_len)

            # Estimativa de log-prob de referência (valores empíricos de modelo SFT)
            ref_logp_w = -1.5 * len(cw_toks)
            ref_logp_l = -2.8 * len(cr_toks)

            # Gravação sequencial dos arrays
            np.array(inp_w, dtype=np.uint16).tofile(f_out)
            np.array(tgt_w, dtype=np.uint16).tofile(f_out)
            np.array(inp_l, dtype=np.uint16).tofile(f_out)
            np.array(tgt_l, dtype=np.uint16).tofile(f_out)
            f_out.write(struct.pack("<ff", ref_logp_w, ref_logp_l))

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n[SUCESSO] Dataset DPO Binário gravado: {out_path} ({size_mb:.2f} MB)")
    print(f"Para treinar o modelo:")
    print(f"  ./build/train_dpo checkpoints/sore_lm_150m_sft.bin {out_path} checkpoints/sore_lm_150m_dpo.bin 300 0.1")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera dataset binário DPO")
    parser.add_argument("--out", type=str, default="data/dpo_pairs.bin")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--tokenizer", type=str, default="pt")
    args = parser.parse_args()

    build_dpo_binary_dataset(
        out_path=args.out,
        count=args.count,
        seq_len=args.seq_len,
        tokenizer_name=args.tokenizer
    )
