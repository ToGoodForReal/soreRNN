#!/usr/bin/env python3
"""
Pipeline de Dados soreRNN-LM: Pré-Treinamento (com Interleaving e Validação Held-Out)
e Ajuste Fino Supervisionado (SFT com Máscara de Loss Cirúrgica).
"""

import os
import sys
import argparse
import random
import numpy as np
from datasets import load_dataset

from tokenizer_utils import get_tokenizer
from prepare_4b_pretrain import generate_dataset

def download_and_stream_sft(
    out_path: str = "data/sft_chat_pt.bin",
    val_out_path: str = "data/sft_val.bin",
    target_tokens: int = 2_000_000,
    val_tokens: int = 100_000,
    tokenizer_name: str = "pt"
):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if val_out_path:
        os.makedirs(os.path.dirname(os.path.abspath(val_out_path)), exist_ok=True)

    tok = get_tokenizer(tokenizer_name)
    eos_id = tok.eos_token_id

    print(f"\n[SFT Data] Carregando diálogos instrucionais em português (Alpaca-PT / Dolly-PT)...")
    chat_sources = [
        {"name": "dominguesm/alpaca-data-pt-br", "split": "train"},
        {"name": "databricks/databricks-dolly-15k", "split": "train"}
    ]

    all_dialogs = []
    for src in chat_sources:
        try:
            ds = load_dataset(src["name"], split=src["split"])
            for row in ds:
                inst = str(row.get("instruction") or row.get("pergunta") or row.get("prompt") or "").strip()
                inp = str(row.get("input") or row.get("contexto") or "").strip()
                out = str(row.get("output") or row.get("resposta") or row.get("response") or "").strip()
                if not inst or not out:
                    continue
                user_msg = f"{inst}\n{inp}".strip() if inp else inst
                prompt = f"<|user|>\n{user_msg}\n<|assistant|>\n"
                resp = f"{out}\n<|endoftext|>\n"
                all_dialogs.append((prompt, resp))
        except Exception as e:
            print(f" -> Aviso ao carregar {src['name']}: {e}", file=sys.stderr)

    if not all_dialogs:
        print("[Erro] Nenhum diálogo carregado para SFT.", file=sys.stderr)
        return

    random.seed(42)
    random.shuffle(all_dialogs)

    # Separação estrita de validação (10% ou até val_tokens)
    split_idx = min(len(all_dialogs) // 10, 2000)
    val_dialogs = all_dialogs[:split_idx]
    train_dialogs = all_dialogs[split_idx:]

    def write_sft_bin(dialogs, filepath, max_tokens):
        with open(filepath, "wb") as f_out:
            written = 0
            buf = []
            idx = 0
            while written < max_tokens:
                if idx >= len(dialogs):
                    random.shuffle(dialogs)
                    idx = 0
                prompt, resp = dialogs[idx]
                idx += 1

                p_toks = tok.encode(prompt, allowed_special={"<|endoftext|>"})
                r_toks = tok.encode(resp, allowed_special={"<|endoftext|>"})
                seq = p_toks + r_toks

                for i in range(len(seq) - 1):
                    inp = seq[i]
                    tgt = 65535 if i < len(p_toks) - 1 else seq[i + 1]
                    buf.append(inp)
                    buf.append(tgt)
                    written += 1
                    if written >= max_tokens:
                        break

                if len(buf) >= 100_000:
                    arr = np.array(buf, dtype=np.uint16)
                    arr.tofile(f_out)
                    f_out.flush()
                    buf.clear()

            if buf:
                arr = np.array(buf, dtype=np.uint16)
                arr.tofile(f_out)
                f_out.flush()
        print(f" -> Arquivo SFT gerado: {filepath} ({written:,} tokens mascarados).")

    if val_out_path and val_tokens > 0:
        write_sft_bin(val_dialogs, val_out_path, val_tokens)
    write_sft_bin(train_dialogs, out_path, target_tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara datasets em streaming para o soreRNN-LM")
    parser.add_argument("--out_pretrain", type=str, default="data/pretrain_75pt_25en.bin")
    parser.add_argument("--val_pretrain", type=str, default="data/val.bin")
    parser.add_argument("--out_sft", type=str, default="data/sft_chat_pt.bin")
    parser.add_argument("--val_sft", type=str, default="data/sft_val.bin")
    parser.add_argument("--tokens_pretrain", type=int, default=20_000_000)
    parser.add_argument("--tokens_val", type=int, default=500_000)
    parser.add_argument("--tokens_sft", type=int, default=2_000_000)
    parser.add_argument("--tokenizer", type=str, default="pt", choices=["pt", "gpt2", "llama"])
    parser.add_argument("--target_4b", action="store_true", help="Alvo de 4 Bilhões de tokens")
    args = parser.parse_args()

    pretrain_target = 4_000_000_000 if args.target_4b else args.tokens_pretrain

    generate_dataset(
        out_path=args.out_pretrain,
        val_out_path=args.val_pretrain,
        target_tokens=pretrain_target,
        val_tokens=args.tokens_val,
        tokenizer_name=args.tokenizer,
        pt_weight=0.80,
        edu_weight=0.10,
        math_weight=0.10,
        paired_mode=True
    )

    download_and_stream_sft(
        out_path=args.out_sft,
        val_out_path=args.val_sft,
        target_tokens=args.tokens_sft,
        val_tokens=100_000,
        tokenizer_name=args.tokenizer
    )
