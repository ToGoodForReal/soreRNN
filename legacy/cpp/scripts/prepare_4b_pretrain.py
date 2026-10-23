#!/usr/bin/env python3
"""
soreRNN-LM: Pipeline de Pré-Treinamento em Escala Industrial (4B Tokens)
Dynamic Fine-Grained Document Interleaving com Deduplicação MinHash e Validação Held-Out

Distribuição Refatorada da Mistura:
  - 80% Português Enriquecido (Wikipedia-PT, C4-PT, Madras1 PT-BR)
  - 10% Educação e Ciência de Alto Valor (FineWeb-Edu com filtro score >= 3.0)
  - 10% Matemática Real + Raciocínio Procedural Variado (MetaMathQA + Alta Entropia)

Melhorias Críticas de Qualidade:
  1. Deduplicação MinHash LSH (64 hashes) e exata (xxhash).
  2. Interleaving de grão fino por documento (elimina blocos monolíticos de 500k tokens).
  3. Fronteira explícita de documento (EOS) com máscara de transição (target=65535).
  4. Geração garantida de val.bin separado com dedup estrito (zero vazamento de validação).
  5. Tokenizer nativo em português (pierreguillou/gpt2-small-portuguese) para máxima densidade.
"""

import os
import sys
import time
import random
import argparse
import numpy as np
from datasets import load_dataset

from tokenizer_utils import get_tokenizer
from dedup_filter import MinHashDeduplicator, filter_fineweb_edu, filter_text_quality
from math_dataset_utils import balanced_math_stream

def hf_stream_pt(sources, dedup: MinHashDeduplicator, text_keys=["text", "conteudo"]):
    """Stream de dados em português com verificação heurística e deduplicação."""
    while True:
        for src in sources:
            name = src["name"]
            subset = src.get("subset")
            split = src.get("split", "train")
            try:
                ds = load_dataset(name, subset, split=split, streaming=True) if subset else load_dataset(name, split=split, streaming=True)
                for row in ds:
                    text = ""
                    for k in text_keys:
                        if k in row and row[k]:
                            text = str(row[k])
                            break
                    if not text:
                        continue

                    # Filtro de qualidade e heurística de PT
                    if not filter_text_quality(text, min_chars=150, is_pt=True):
                        continue

                    yield text.strip()
            except Exception as e:
                print(f" -> [Aviso Stream PT] Fonte {name}: {e}. Alternando...", file=sys.stderr)
                time.sleep(2)

def hf_stream_edu(sources, dedup: MinHashDeduplicator, text_keys=["text"]):
    """Stream de dados educacionais (FineWeb-Edu) com filtro de pontuação educacional."""
    while True:
        for src in sources:
            name = src["name"]
            subset = src.get("subset")
            split = src.get("split", "train")
            try:
                ds = load_dataset(name, subset, split=split, streaming=True) if subset else load_dataset(name, split=split, streaming=True)
                for row in ds:
                    if not filter_fineweb_edu(row, min_score=3.0):
                        continue

                    text = ""
                    for k in text_keys:
                        if k in row and row[k]:
                            text = str(row[k])
                            break
                    if not text:
                        continue

                    if not filter_text_quality(text, min_chars=150, is_pt=False):
                        continue

                    yield text.strip()
            except Exception as e:
                print(f" -> [Aviso Stream Edu] Fonte {name}: {e}. Alternando...", file=sys.stderr)
                time.sleep(2)

def build_paired_sequence(tokens: list, eos_id: int) -> list:
    """
    Constrói a sequência emparelhada [input_0, target_0, input_1, target_1, ...]
    com fronteira de documento:
    - Para tokens dentro do documento: target[i] = input[i+1]
    - Para o token EOS final: target = 65535 (IGNORE_INDEX)
    Garante que o modelo NUNCA tenta prever o início do próximo documento não relacionado.
    """
    seq = list(tokens)
    if not seq or seq[-1] != eos_id:
        seq.append(eos_id)

    pairs = []
    L = len(seq)
    for i in range(L - 1):
        pairs.append(seq[i])
        pairs.append(seq[i + 1])
    # Último token (EOS): mascarado com 65535
    pairs.append(seq[-1])
    pairs.append(65535)
    return pairs

def generate_dataset(
    out_path: str,
    val_out_path: str,
    target_tokens: int,
    val_tokens: int,
    tokenizer_name: str = "pt",
    pt_weight: float = 0.80,
    edu_weight: float = 0.10,
    math_weight: float = 0.10,
    paired_mode: bool = True
):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if val_out_path:
        os.makedirs(os.path.dirname(os.path.abspath(val_out_path)), exist_ok=True)

    tok = get_tokenizer(tokenizer_name)
    eos_id = tok.eos_token_id
    dedup = MinHashDeduplicator()

    pt_sources = [
        {"name": "wikimedia/wikipedia", "subset": "20231101.pt", "split": "train"},
        {"name": "allenai/c4", "subset": "pt", "split": "train"},
        {"name": "Madras1/corpus-ptbr-v1", "subset": None, "split": "train"},
    ]

    en_edu_sources = [
        {"name": "HuggingFaceFW/fineweb-edu", "subset": "sample-10BT", "split": "train"}
    ]

    pt_gen = hf_stream_pt(pt_sources, dedup)
    edu_gen = hf_stream_edu(en_edu_sources, dedup)
    math_gen = balanced_math_stream(real_prob=0.5)

    domains = ["pt", "edu", "math"]
    weights = [pt_weight, edu_weight, math_weight]
    generators = {"pt": pt_gen, "edu": edu_gen, "math": math_gen}

    print("=" * 80)
    print("       soreRNN-LM: GERADOR DE PRÉ-TREINO COM DEDUP E INTERLEAVING       ")
    print("=" * 80)
    print(f" Tokenizer:         {tokenizer_name} (vocab={tok.vocab_size}, eos_id={eos_id})")
    print(f" Saída Treino:      {out_path} ({target_tokens:,} tokens)")
    print(f" Saída Validação:   {val_out_path} ({val_tokens:,} tokens)")
    print(f" Modo Emparelhado:  {'Sim (Input + Target Mascarado em 65535)' if paired_mode else 'Sequencial Simples'}")
    print(" Distribuição:")
    print(f"   - Português:     {pt_weight * 100:.1f}% (Wikipedia, C4, Madras1)")
    print(f"   - Educação/Edu:  {edu_weight * 100:.1f}% (FineWeb-Edu score >= 3.0)")
    print(f"   - Matemática:    {math_weight * 100:.1f}% (MetaMathQA Real + Procedural Variado)")
    print("=" * 80)

    # 1. FASE 1: Geração do Dataset de Validação Held-Out (com Deduplicação Rígida)
    if val_out_path and val_tokens > 0 and (not os.path.exists(val_out_path) or os.path.getsize(val_out_path) < (val_tokens * (4 if paired_mode else 2))):
        print(f"\n[Fase 1/2] Gerando {val_tokens:,} tokens de Validação Held-Out ({val_out_path})...")
        val_written = 0
        with open(val_out_path, "wb") as f_val:
            while val_written < val_tokens:
                chosen_domain = random.choices(domains, weights=weights)[0]
                text = next(generators[chosen_domain])

                # Adiciona à validação (rejeita duplicata interna da val)
                if not dedup.add_validation_document(text):
                    continue

                toks = tok.encode(text)
                if paired_mode:
                    chunk = build_paired_sequence(toks, eos_id)
                    arr = np.array(chunk, dtype=np.uint16)
                    arr.tofile(f_val)
                    val_written += len(chunk) // 2
                else:
                    toks.append(eos_id)
                    arr = np.array(toks, dtype=np.uint16)
                    arr.tofile(f_val)
                    val_written += len(toks)

        print(f" -> Validação Held-Out concluída: {val_written:,} tokens ({len(dedup.val_signatures)} documentos indexados).")
    else:
        print(f"\n[Fase 1/2] Validação held-out já existe ou desabilitada ({val_out_path}).")

    # 2. FASE 2: Geração do Dataset de Treinamento (com Verificação Estrita Anti-Vazamento)
    bytes_per_token = 4 if paired_mode else 2
    existing_tokens = 0
    if os.path.exists(out_path):
        existing_tokens = os.path.getsize(out_path) // bytes_per_token
        print(f"\n[Resume] Arquivo existente detectado com {existing_tokens:,} tokens.")

    if existing_tokens >= target_tokens:
        print(f"Meta de {target_tokens:,} tokens já atingida em {out_path}. Concluído!")
        return

    print(f"\n[Fase 2/2] Gerando Treino com Interleaving por Documento ({target_tokens - existing_tokens:,} tokens restantes)...")
    total_written = existing_tokens
    domain_counts = {"pt": int(existing_tokens * pt_weight), "edu": int(existing_tokens * edu_weight), "math": int(existing_tokens * math_weight)}
    rejected_dups = 0
    rejected_val_leaks = 0

    mode = "ab" if existing_tokens > 0 else "wb"
    t_start = time.time()
    t_last = time.time()
    tokens_since_last = 0

    buffer_chunk = []
    flush_token_count = 100_000

    with open(out_path, mode) as f_out:
        while total_written < target_tokens:
            chosen_domain = random.choices(domains, weights=weights)[0]
            text = next(generators[chosen_domain])

            # Deduplicação: verifica vazamento de validação e duplicata de treino
            if dedup.is_val_leak(text):
                rejected_val_leaks += 1
                continue

            if not dedup.add_train_document(text):
                rejected_dups += 1
                continue

            toks = tok.encode(text)
            if paired_mode:
                seq_pairs = build_paired_sequence(toks, eos_id)
                num_new_tokens = len(seq_pairs) // 2
                buffer_chunk.extend(seq_pairs)
            else:
                toks.append(eos_id)
                num_new_tokens = len(toks)
                buffer_chunk.extend(toks)

            domain_counts[chosen_domain] += num_new_tokens
            total_written += num_new_tokens
            tokens_since_last += num_new_tokens

            if len(buffer_chunk) >= (flush_token_count * (2 if paired_mode else 1)) or total_written >= target_tokens:
                arr = np.array(buffer_chunk, dtype=np.uint16)
                arr.tofile(f_out)
                f_out.flush()
                buffer_chunk.clear()

                now = time.time()
                elapsed = max(now - t_last, 0.001)
                if elapsed >= 5.0 or total_written >= target_tokens:
                    speed = tokens_since_last / elapsed
                    speed_global = (total_written - existing_tokens) / max(now - t_start, 0.001)
                    pct = (total_written / target_tokens) * 100.0
                    rem_sec = (target_tokens - total_written) / max(speed_global, 1.0)
                    eta_h = rem_sec / 3600.0

                    pt_p = (domain_counts["pt"] / max(total_written, 1)) * 100.0
                    edu_p = (domain_counts["edu"] / max(total_written, 1)) * 100.0
                    math_p = (domain_counts["math"] / max(total_written, 1)) * 100.0

                    print(
                        f"[{pct:5.2f}%] Gravados: {total_written:,}/{target_tokens:,} tok | "
                        f"Vel: {speed:,.0f} tok/s | ETA: {eta_h:.1f}h | "
                        f"Mix: PT {pt_p:.1f}% | Edu {edu_p:.1f}% | Math {math_p:.1f}% | "
                        f"Dups Rejeitadas: {rejected_dups:,} | Vazamentos Val: {rejected_val_leaks:,}"
                    )
                    t_last = now
                    tokens_since_last = 0

    print("\n" + "=" * 80)
    print(f" [SUCESSO COMPLETO] Dataset de Pré-Treinamento Concluído:")
    print(f" Treino:     {out_path} ({total_written:,} tokens, {os.path.getsize(out_path)/(1024**3):.2f} GB)")
    if val_out_path and os.path.exists(val_out_path):
        print(f" Validação:  {val_out_path} ({os.path.getsize(val_out_path)/(1024**2):.1f} MB)")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Pré-Treino soreRNN-LM 4B")
    parser.add_argument("--out", type=str, default="data/pretrain_4b.bin")
    parser.add_argument("--val_out", type=str, default="data/val.bin")
    parser.add_argument("--tokens", type=int, default=4_000_000_000)
    parser.add_argument("--val_tokens", type=int, default=1_000_000)
    parser.add_argument("--tokenizer", type=str, default="pt", choices=["pt", "gpt2", "llama"])
    parser.add_argument("--pt_weight", type=float, default=0.80)
    parser.add_argument("--edu_weight", type=float, default=0.10)
    parser.add_argument("--math_weight", type=float, default=0.10)
    parser.add_argument("--paired", action="store_true", default=True, help="Grava no formato emparelhado com máscara de fronteira (target=65535)")
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
