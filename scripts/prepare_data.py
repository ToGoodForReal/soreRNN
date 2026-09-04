#!/usr/bin/env python3
"""
Script de Download, Filtragem (75% PT-BR / 25% EN) e Tokenização de Dados
para o Framework soreRNN (C++20/CUDA).
"""

import os
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset

def get_tokenizer():
    return tiktoken.get_encoding("gpt2")

def download_and_tokenize(
    out_pretrain_bin: str,
    out_sft_bin: str,
    target_pretrain_tokens: int = 200_000,
    target_sft_tokens: int = 50_000
):
    enc = get_tokenizer()
    os.makedirs(os.path.dirname(out_pretrain_bin), exist_ok=True)
    os.makedirs(os.path.dirname(out_sft_bin), exist_ok=True)

    print("=" * 65)
    print("  PREPARAÇÃO DE DADOS: 75% PT-BR / 25% EN + SFT CONVERSACIONAL")
    print("=" * 65)
    print(f"Alvo Pré-treino: {target_pretrain_tokens:,} tokens (75% PT / 25% EN)")
    print(f"Alvo SFT Conversa: {target_sft_tokens:,} tokens")

    pt_token_goal = int(target_pretrain_tokens * 0.75)
    en_token_goal = int(target_pretrain_tokens * 0.25)

    pt_tokens = []
    en_tokens = []
    sft_tokens = []

    # 1. Obtenção de Dados em Português (75%)
    print("\n[1/3] Baixando textos em Português (PT-BR)...")
    try:
        # Tenta carregar Wikipedia em português ou fallback para textos ricos em PT
        ds_pt = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True)
        for row in ds_pt:
            text = row.get("text", "")
            if text:
                toks = enc.encode(text)
                pt_tokens.extend(toks)
                if len(pt_tokens) >= pt_token_goal:
                    break
        print(f" -> Sucesso! Coletados {len(pt_tokens):,} tokens em PT-BR.")
    except Exception as e:
        print(f" -> Aviso ao conectar com dataset PT ({e}). Usando gerador de contingência PT-BR.")
        sample_pt_corpus = (
            "A inteligência artificial e os modelos de linguagem avançaram significativamente. "
            "A arquitetura de redes neurais recorrentes lineares, como o Griffin e o RG-LRU do Google, "
            "permitem processamento em tempo O(1) durante a inferência e paralelização total no treino. "
            "O Brasil possui uma rica diversidade cultural, científica e tecnológica. "
            "O processamento de linguagem natural em português requer atenção a nuances de concordância e conjugação. "
            "Modelos com estado oculto linear conseguem manter memória contextual através de portões de decaimento reais. "
        ) * 5000
        pt_tokens = enc.encode(sample_pt_corpus)[:pt_token_goal]

    # 2. Obtenção de Dados em Inglês de Alta Qualidade (25% - Raciocínio/Lógica)
    print("\n[2/3] Baixando textos em Inglês (25% - Lógica & Raciocínio)...")
    try:
        ds_en = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        for row in ds_en:
            text = row.get("text", "")
            if text:
                toks = enc.encode(text)
                en_tokens.extend(toks)
                if len(en_tokens) >= en_token_goal:
                    break
        print(f" -> Sucesso! Coletados {len(en_tokens):,} tokens em Inglês.")
    except Exception as e:
        print(f" -> Aviso ao conectar com dataset EN ({e}). Usando gerador de contingência EN.")
        sample_en_corpus = (
            "Recurrent neural networks with linear state updates enable fast associative memory. "
            "Parallel associative scan reduces the sequential complexity to logarithmic time on GPUs. "
            "Logic, mathematics, and structured reasoning form the foundation of language modeling. "
        ) * 3000
        en_tokens = enc.encode(sample_en_corpus)[:en_token_goal]

    # 3. Intercalação 75% PT-BR / 25% EN
    print("\n[Mix] Mesclando corpus: 75% Português + 25% Inglês...")
    pretrain_all = []
    chunk_pt = 300
    chunk_en = 100

    i_pt = 0
    i_en = 0
    while len(pretrain_all) < target_pretrain_tokens and (i_pt < len(pt_tokens) or i_en < len(en_tokens)):
        if i_pt < len(pt_tokens):
            take = min(chunk_pt, len(pt_tokens) - i_pt)
            pretrain_all.extend(pt_tokens[i_pt : i_pt + take])
            i_pt += take
        if i_en < len(en_tokens):
            take = min(chunk_en, len(en_tokens) - i_en)
            pretrain_all.extend(en_tokens[i_en : i_en + take])
            i_en += take

    pretrain_arr = np.array(pretrain_all[:target_pretrain_tokens], dtype=np.uint16)
    pretrain_arr.tofile(out_pretrain_bin)
    print(f" -> Arquivo de Pré-treino salvo: {out_pretrain_bin} ({len(pretrain_arr):,} tokens, {os.path.getsize(out_pretrain_bin)/(1024*1024):.2f} MB)")

    # 4. Obtenção do Dataset de SFT Conversacional (Instruções em PT-BR)
    print("\n[3/3] Baixando dataset de SFT Conversacional (Cabrita / Alpaca PT-BR)...")
    try:
        ds_sft = load_dataset("gustavor/cabrita", split="train", streaming=True)
        for row in ds_sft:
            instruction = row.get("instruction", "")
            inp = row.get("input", "")
            output = row.get("output", "")

            user_msg = f"{instruction} {inp}".strip()
            # Formatação canônica de chat: <|user|> ... <|assistant|> ...
            formatted_dialog = f"<|user|>\n{user_msg}\n<|assistant|>\n{output}\n<|endoftext|>\n"
            toks = enc.encode(formatted_dialog, allowed_special={"<|endoftext|>"})
            sft_tokens.extend(toks)

            if len(sft_tokens) >= target_sft_tokens:
                break
        print(f" -> Sucesso! Coletados {len(sft_tokens):,} tokens de diálogo em PT-BR.")
    except Exception as e:
        print(f" -> Aviso ao conectar com dataset SFT ({e}). Usando diálogos de contingência.")
        dialogs = [
            "<|user|>\nOlá, quem é você?\n<|assistant|>\nOlá! Sou uma inteligência artificial treinada do zero em C++ e CUDA com arquitetura de RNN Linear.\n<|endoftext|>\n",
            "<|user|>\nO que é uma RNN Linear?\n<|assistant|>\nUma RNN Linear é um modelo recorrente onde a transição de estados oculta é estritamente linear, permitindo paralelização via associative scan no treino e altíssima velocidade na inferência.\n<|endoftext|>\n",
            "<|user|>\nExplique a vantagem do C++ em IA.\n<|assistant|>\nO C++ com CUDA oferece controle direto sobre alinhamento de memória física, eliminação de overheads de garbage collection e saturação da largura de banda da GPU.\n<|endoftext|>\n"
        ] * 1000
        for d in dialogs:
            sft_tokens.extend(enc.encode(d, allowed_special={"<|endoftext|>"}))
            if len(sft_tokens) >= target_sft_tokens:
                break

    sft_arr = np.array(sft_tokens[:target_sft_tokens], dtype=np.uint16)
    sft_arr.tofile(out_sft_bin)
    print(f" -> Arquivo de SFT Conversacional salvo: {out_sft_bin} ({len(sft_arr):,} tokens, {os.path.getsize(out_sft_bin)/(1024*1024):.2f} MB)")
    print("\n[OK] Pipeline de dados concluído com sucesso!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara datasets para o soreRNN")
    parser.add_argument("--out_pretrain", type=str, default="data/pretrain_75pt_25en.bin")
    parser.add_argument("--out_sft", type=str, default="data/sft_chat_pt.bin")
    parser.add_argument("--tokens_pretrain", type=int, default=200_000)
    parser.add_argument("--tokens_sft", type=int, default=50_000)
    args = parser.parse_args()

    download_and_tokenize(
        out_pretrain_bin=args.out_pretrain,
        out_sft_bin=args.out_sft,
        target_pretrain_tokens=args.tokens_pretrain,
        target_sft_tokens=args.tokens_sft
    )
