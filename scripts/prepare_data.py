#!/usr/bin/env python3
"""
Pipeline de Dados com Curriculum de Alta Densidade (Textbooks-Style)
para o soreRNN-LM (152M Parâmetros).

Composição:
- 40% Português Enriquecido (Wikipedia PT, C4-PT, Madras1)
- 30% Conhecimento Geral e Ciência (FineWeb-Edu, TinyStories)
- 20% Matemática e Raciocínio com Cadeia de Pensamento (Chain-of-Thought)
- 10% SFT Conversacional (Alpaca-PT, BPT-SFT)

Suporta streaming contínuo em disco para até 4 Bilhões de tokens.
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset

def get_tokenizer():
    return tiktoken.get_encoding("gpt2")

def generate_math_reasoning_samples(target_tokens: int, enc, out_file) -> int:
    """Gera dados de matemática e raciocínio lógico em português com passo a passo."""
    print(f"\n[Curriculum] Gerando {target_tokens:,} tokens de Matemática e Raciocínio CoT...")
    collected = 0
    buffer = []
    chunk_flush_size = 150_000

    operations = ["soma", "subtracao", "multiplicacao", "divisao", "problema_logico"]

    while collected < target_tokens:
        op = random.choice(operations)
        text = ""

        if op == "soma":
            a = random.randint(1, 999)
            b = random.randint(1, 999)
            ans = a + b
            text = (
                f"Problema: Calcule {a} + {b}.\n"
                f"Raciocínio: Para somar {a} e {b}, decompomos os valores: "
                f"({a // 10 * 10} + {b // 10 * 10}) + ({a % 10} + {b % 10}) = {ans}.\n"
                f"Portanto, {a} + {b} = {ans}. Resposta: {ans}.\n\n"
            )
        elif op == "subtracao":
            a = random.randint(10, 999)
            b = random.randint(1, a)
            ans = a - b
            text = (
                f"Problema: Quanto é {a} menos {b}?\n"
                f"Raciocínio: Subtraindo {b} de {a}, obtemos a diferença exata de {ans}.\n"
                f"Resposta: {ans}.\n\n"
            )
        elif op == "multiplicacao":
            a = random.randint(2, 50)
            b = random.randint(2, 20)
            ans = a * b
            text = (
                f"Problema: Quanto é {a} multiplicado por {b}?\n"
                f"Raciocínio: {a} vezes {b} equivale a somar {a} repetido {b} vezes, totalizando {ans}.\n"
                f"Resposta: {ans}.\n\n"
            )
        elif op == "divisao":
            b = random.randint(2, 12)
            ans = random.randint(1, 30)
            a = b * ans
            text = (
                f"Problema: Divida {a} por {b}.\n"
                f"Raciocínio: Qual número multiplicado por {b} resulta em {a}? Esse número é {ans}, pois {ans} x {b} = {a}.\n"
                f"Resposta: {ans}.\n\n"
            )
        elif op == "problema_logico":
            itens = ["maçãs", "livros", "canetas", "moedas", "figurinhas"]
            item = random.choice(itens)
            n1 = random.randint(5, 50)
            n2 = random.randint(2, n1 - 1)
            sobra = n1 - n2
            text = (
                f"Problema: Lucas tinha {n1} {item} e deu {n2} para seu amigo. Com quantas {item} Lucas ficou?\n"
                f"Raciocínio: Começando com {n1}, subtraímos as {n2} que ele deu: {n1} - {n2} = {sobra}.\n"
                f"Resposta: Lucas ficou com {sobra} {item}.\n\n"
            )

        toks = enc.encode(text)
        buffer.extend(toks)
        collected += len(toks)

        if len(buffer) >= chunk_flush_size:
            arr = np.array(buffer, dtype=np.uint16)
            arr.tofile(out_file)
            out_file.flush()
            buffer.clear()
            print(f"    [Math CoT] {collected:,} / {target_tokens:,} tokens gravados...")

    if buffer:
        arr = np.array(buffer, dtype=np.uint16)
        arr.tofile(out_file)
        out_file.flush()
        buffer.clear()

    print(f" -> Sucesso: {collected:,} tokens de raciocínio matemático gerados!")
    return collected

def stream_dataset_to_file(
    dataset_name: str,
    subset: str,
    split: str,
    text_keys: list,
    out_file,
    target_tokens: int,
    enc,
    lang_label: str
) -> int:
    print(f" -> Conectando a '{dataset_name}' (subset={subset})...")
    collected = 0
    buffer = []
    chunk_flush_size = 200_000

    try:
        if subset:
            ds = load_dataset(dataset_name, subset, split=split, streaming=True)
        else:
            ds = load_dataset(dataset_name, split=split, streaming=True)

        t0 = time.time()
        for row in ds:
            text = ""
            for k in text_keys:
                if k in row and row[k]:
                    text = row[k]
                    break

            if not text or len(text.strip()) < 30:
                continue

            toks = enc.encode(text)
            buffer.extend(toks)
            collected += len(toks)

            if len(buffer) >= chunk_flush_size:
                arr = np.array(buffer, dtype=np.uint16)
                arr.tofile(out_file)
                out_file.flush()
                buffer.clear()

                elapsed = max(time.time() - t0, 0.01)
                speed = collected / elapsed
                print(f"    [{lang_label}] {collected:,} / {target_tokens:,} tokens ({speed:,.0f} tok/s)...")

            if collected >= target_tokens:
                break

        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            arr.tofile(out_file)
            out_file.flush()
            buffer.clear()

        print(f" -> Sucesso com '{dataset_name}': {collected:,} tokens coletados.")
        return collected

    except Exception as e:
        print(f" -> Erro ao conectar a '{dataset_name}': {e}")
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            arr.tofile(out_file)
            out_file.flush()
        return collected

def download_and_stream_pretrain(
    out_path: str,
    target_tokens: int = 4_000_000_000
):
    enc = get_tokenizer()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Curriculum de Alta Densidade
    pt_target = int(target_tokens * 0.45)   # 45% Português
    math_target = int(target_tokens * 0.25) # 25% Matemática / CoT
    en_target = target_tokens - pt_target - math_target # 30% Ciência / Inglês

    print("=" * 70)
    print("  soreRNN-LM v2: CURRICULUM DE ALTA DENSIDADE (ALVO: 4B TOKENS)")
    print("=" * 70)
    print(f" Arquivo de Saída: {out_path}")
    print(f" Total de Tokens Alvo: {target_tokens:,} (~{target_tokens * 2 / (1024**3):.2f} GB)")
    print(f"   - Português Enriquecido (45%): {pt_target:,} tokens")
    print(f"   - Matemática & Raciocínio CoT (25%): {math_target:,} tokens")
    print(f"   - Ciência & Conhecimento Geral (30%): {en_target:,} tokens")
    print("=" * 70)

    pt_sources = [
        {"name": "wikimedia/wikipedia", "subset": "20231101.pt", "split": "train", "keys": ["text"]},
        {"name": "allenai/c4", "subset": "pt", "split": "train", "keys": ["text"]},
        {"name": "Madras1/corpus-ptbr-v1", "subset": None, "split": "train", "keys": ["text", "conteudo"]},
    ]

    en_sources = [
        {"name": "HuggingFaceFW/fineweb-edu", "subset": "sample-10BT", "split": "train", "keys": ["text"]},
        {"name": "roneneldan/TinyStories", "subset": None, "split": "train", "keys": ["text"]},
    ]

    with open(out_path, "wb") as f_out:
        # 1. Matemática e Raciocínio CoT
        math_done = generate_math_reasoning_samples(math_target, enc, f_out)

        # 2. Português
        print("\n[Fase 2/3] Iniciando streaming de corpus em Português (PT-BR)...")
        pt_done = 0
        for src in pt_sources:
            if pt_done >= pt_target:
                break
            rem = pt_target - pt_done
            got = stream_dataset_to_file(
                src["name"], src["subset"], src["split"], src["keys"],
                f_out, rem, enc, "PT-BR"
            )
            pt_done += got

        # 3. Ciência e Conhecimento Geral
        print("\n[Fase 3/3] Iniciando streaming de Ciência / Conhecimento Geral...")
        en_done = 0
        for src in en_sources:
            if en_done >= en_target:
                break
            rem = en_target - en_done
            got = stream_dataset_to_file(
                src["name"], src["subset"], src["split"], src["keys"],
                f_out, rem, enc, "Ciência/EN"
            )
            en_done += got

    total_all = math_done + pt_done + en_done
    final_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print("\n" + "=" * 70)
    print(f" [OK] Pré-Treino Gravado com Sucesso: {total_all:,} tokens ({final_size_mb:.2f} MB)")
    print("=" * 70)

def download_and_stream_sft(
    out_path: str,
    target_tokens: int = 5_000_000
):
    enc = get_tokenizer()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("\n" + "=" * 70)
    print("  soreRNN-LM: SFT CONVERSACIONAL COM DIÁLOGOS DIVERSIFICADOS")
    print("=" * 70)

    sft_sources = [
        {"name": "dominguesm/alpaca-data-pt-br", "split": "train"},
        {"name": "recogna-nlp/bpt_sft_dataset", "split": "train"},
    ]

    total_tokens = 0
    dialog_count = 0
    buffer = []

    with open(out_path, "wb") as f_out:
        for src in sft_sources:
            src_name = src["name"]
            print(f" -> Conectando ao dataset SFT '{src_name}'...")
            try:
                ds = load_dataset(src_name, split=src["split"], streaming=True)
                for row in ds:
                    instruction = row.get("instruction", "") or row.get("pergunta", "") or row.get("prompt", "")
                    inp = row.get("input", "") or row.get("contexto", "")
                    output = row.get("output", "") or row.get("resposta", "") or row.get("completion", "")

                    if not instruction or not output:
                        continue

                    user_msg = f"{instruction} {inp}".strip()
                    dialog = f"<|user|>\n{user_msg}\n<|assistant|>\n{output}\n<|endoftext|>\n"
                    toks = enc.encode(dialog, allowed_special={"<|endoftext|>"})

                    buffer.extend(toks)
                    total_tokens += len(toks)
                    dialog_count += 1

                    if len(buffer) >= 100_000:
                        arr = np.array(buffer, dtype=np.uint16)
                        arr.tofile(f_out)
                        f_out.flush()
                        buffer.clear()
                        print(f"    [SFT] {total_tokens:,} tokens ({dialog_count:,} diálogos salvos)...")

                    if total_tokens >= target_tokens:
                        break

                if buffer:
                    arr = np.array(buffer, dtype=np.uint16)
                    arr.tofile(f_out)
                    f_out.flush()
                    buffer.clear()

                if total_tokens >= target_tokens:
                    break
            except Exception as e:
                print(f" -> Aviso ao carregar '{src_name}': {e}. Tentando próximo candidato...")

    print(f"\n [OK] SFT Concluído: {total_tokens:,} tokens gravados ({dialog_count:,} diálogos reais).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara datasets em streaming para o soreRNN-LM")
    parser.add_argument("--out_pretrain", type=str, default="data/pretrain_75pt_25en.bin")
    parser.add_argument("--out_sft", type=str, default="data/sft_chat_pt.bin")
    parser.add_argument("--tokens_pretrain", type=int, default=20_000_000)
    parser.add_argument("--tokens_sft", type=int, default=2_000_000)
    parser.add_argument("--target_4b", action="store_true", help="Alvo de 4 Bilhões de tokens")
    args = parser.parse_args()

    pretrain_target = 4_000_000_000 if args.target_4b else args.tokens_pretrain

    download_and_stream_pretrain(args.out_pretrain, target_tokens=pretrain_target)
    download_and_stream_sft(args.out_sft, target_tokens=args.tokens_sft)
