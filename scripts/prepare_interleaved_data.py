#!/usr/bin/env python3
"""
Pipeline de Pré-Treinamento com Interleaving Contínuo de Dados (Dynamic Interleaving).

Em vez de gravar grandes blocos monolíticos sequenciais:
    [1B Math] -> [1.8B Wikipedia] -> [1.2B FineWeb]  (causa esquecimento catastrófico)

Este script realiza multiplexação e intercalação contínua de streams:
    - 45% Português Enriquecido (Wikipedia PT, C4-PT, Madras1)
    - 25% Matemática e Raciocínio com Cadeia de Pensamento (CoT)
    - 30% Ciência e Conhecimento Geral (FineWeb-Edu, TinyStories)

Garante que o otimizador veja todos os domínios a cada intervalo de tokens,
impedindo o esquecimento e regularizando o espaço latente de representações.
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

def math_stream_generator(enc):
    """Gerador contínuo de amostras de raciocínio matemático em português."""
    operations = ["soma", "subtracao", "multiplicacao", "divisao", "problema_logico"]
    itens = ["maçãs", "livros", "canetas", "moedas", "figurinhas", "balas"]

    while True:
        op = random.choice(operations)
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
            item = random.choice(itens)
            n1 = random.randint(5, 50)
            n2 = random.randint(2, n1 - 1)
            sobra = n1 - n2
            text = (
                f"Problema: Lucas tinha {n1} {item} e deu {n2} para seu amigo. Com quantas {item} Lucas ficou?\n"
                f"Raciocínio: Começando com {n1}, subtraímos as {n2} que ele deu: {n1} - {n2} = {sobra}.\n"
                f"Resposta: Lucas ficou com {sobra} {item}.\n\n"
            )

        yield enc.encode(text)

def dataset_stream_generator(sources, enc, text_keys=["text", "conteudo"]):
    """Gerador contínuo que itera sobre múltiplos datasets via streaming do HuggingFace."""
    while True:
        for src in sources:
            name = src["name"]
            subset = src.get("subset")
            split = src.get("split", "train")
            try:
                if subset:
                    ds = load_dataset(name, subset, split=split, streaming=True)
                else:
                    ds = load_dataset(name, split=split, streaming=True)

                for row in ds:
                    text = ""
                    for k in text_keys:
                        if k in row and row[k]:
                            text = row[k]
                            break
                    if not text or len(text.strip()) < 30:
                        continue

                    yield enc.encode(text)
            except Exception as e:
                print(f" -> [Stream Aviso] Erro na fonte {name}: {e}. Avançando...")
                time.sleep(1)

def build_interleaved_pretrain(
    out_path: str,
    target_tokens: int = 4_000_000_000,
    chunk_tokens: int = 250_000
):
    enc = get_tokenizer()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=" * 75)
    print("  soreRNN-LM: PRÉ-TREINO COM INTERLEAVING DINÂMICO MULTI-DOMÍNIO")
    print("=" * 75)
    print(f" Arquivo de Saída: {out_path}")
    print(f" Meta Total de Tokens: {target_tokens:,}")
    print(f" Proporções da Mistura:")
    print(f"   - 45% Português Enriquecido (Wikipedia, C4, Madras1)")
    print(f"   - 25% Matemática e Raciocínio CoT")
    print(f"   - 30% Ciência e Conhecimento Geral (FineWeb-Edu, TinyStories)")
    print(f" Tamanho do Chunk Intercalado: {chunk_tokens:,} tokens")
    print("=" * 75)

    pt_sources = [
        {"name": "wikimedia/wikipedia", "subset": "20231101.pt", "split": "train"},
        {"name": "allenai/c4", "subset": "pt", "split": "train"},
        {"name": "Madras1/corpus-ptbr-v1", "subset": None, "split": "train"},
    ]

    en_sources = [
        {"name": "HuggingFaceFW/fineweb-edu", "subset": "sample-10BT", "split": "train"},
        {"name": "roneneldan/TinyStories", "subset": None, "split": "train"},
    ]

    math_gen = math_stream_generator(enc)
    pt_gen = dataset_stream_generator(pt_sources, enc)
    en_gen = dataset_stream_generator(en_sources, enc)

    # Pesos de amostragem por domínio
    domains = ["pt", "math", "en"]
    weights = [0.45, 0.25, 0.30]

    generators = {
        "pt": pt_gen,
        "math": math_gen,
        "en": en_gen,
    }

    counts = {"pt": 0, "math": 0, "en": 0}
    total_written = 0
    t0 = time.time()

    with open(out_path, "wb") as f_out:
        while total_written < target_tokens:
            # Seleciona o domínio proporcionalmente
            chosen_domain = random.choices(domains, weights=weights)[0]
            gen = generators[chosen_domain]

            # Coleta um chunk do domínio selecionado
            chunk_buffer = []
            while len(chunk_buffer) < chunk_tokens and total_written + len(chunk_buffer) < target_tokens:
                tokens = next(gen)
                chunk_buffer.extend(tokens)

            arr = np.array(chunk_buffer, dtype=np.uint16)
            arr.tofile(f_out)
            f_out.flush()

            counts[chosen_domain] += len(arr)
            total_written += len(arr)

            elapsed = max(time.time() - t0, 0.01)
            speed = total_written / elapsed
            pct = (total_written / target_tokens) * 100.0

            pt_pct = (counts["pt"] / total_written) * 100.0
            math_pct = (counts["math"] / total_written) * 100.0
            en_pct = (counts["en"] / total_written) * 100.0

            print(
                f"[{pct:5.1f}%] Gravados: {total_written:,} / {target_tokens:,} tokens "
                f"({speed:,.0f} tok/s) | Mix: PT {pt_pct:.1f}% | Math {math_pct:.1f}% | Edu {en_pct:.1f}%"
            )

    mb_final = os.path.getsize(out_path) / (1024 * 1024)
    print("\n" + "=" * 75)
    print(f" [SUCESSO] Dataset Intercalado Concluído: {total_written:,} tokens ({mb_final:.2f} MB)")
    print(f" Totais por Domínio:")
    print(f"   - Português: {counts['pt']:,} tokens ({(counts['pt']/total_written)*100:.1f}%)")
    print(f"   - Matemática: {counts['math']:,} tokens ({(counts['math']/total_written)*100:.1f}%)")
    print(f"   - Ciência/EN: {counts['en']:,} tokens ({(counts['en']/total_written)*100:.1f}%)")
    print("===========================================================================")
    os._exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera dataset de pré-treino com interleaving contínuo")
    parser.add_argument("--out", type=str, default="data/pretrain_interleaved.bin")
    parser.add_argument("--tokens", type=int, default=20_000_000, help="Total de tokens alvo")
    parser.add_argument("--chunk_size", type=int, default=200_000, help="Tokens por chunk intercalado")
    args = parser.parse_args()

    build_interleaved_pretrain(args.out, args.tokens, args.chunk_size)
