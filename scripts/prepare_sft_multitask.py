#!/usr/bin/env python3
"""
Pipeline de SFT Multi-Tarefa com Máscara de Loss (Loss Masking).

Combina:
- 65% Diálogos de Conversação Geral em Português (Alpaca-PT)
- 25% Diálogos de Matemática com Raciocínio Passo a Passo (CoT)
- 10% Diálogos de Lógica e Dedução em Português

Formato de Saída (Modo Pares para DataLoader):
- Para cada token: (uint16_t input, uint16_t target)
- Tokens do prompt do usuário recebem target = 65535 (IGNORE_INDEX no kernel CUDA)
- Tokens da resposta do assistente recebem o próximo token real
- Garante 100% de gradiente focado na resposta do assistente, sem memorizar prompts!
"""

import os
import random
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset

def get_tokenizer():
    return tiktoken.get_encoding("gpt2")

def generate_math_sft_pairs(count: int = 12000):
    """Gera pares (prompt, response) de matemática com raciocínio CoT."""
    operations = ["soma", "subtracao", "multiplicacao", "divisao"]
    pairs = []

    for _ in range(count):
        op = random.choice(operations)
        if op == "soma":
            a = random.randint(1, 999)
            b = random.randint(1, 999)
            ans = a + b
            q = random.choice([
                f"Calcule {a} + {b}.",
                f"Quanto é {a} mais {b}?",
                f"Qual o resultado da soma de {a} e {b}?",
                f"Some os números {a} e {b} por favor.",
            ])
            r = (
                f"Para calcular {a} + {b}, decompomos os valores: "
                f"({a // 10 * 10} + {b // 10 * 10}) + ({a % 10} + {b % 10}) = {ans}. "
                f"Portanto, {a} + {b} = {ans}. Resposta: {ans}."
            )
        elif op == "subtracao":
            a = random.randint(10, 999)
            b = random.randint(1, a)
            ans = a - b
            q = random.choice([
                f"Quanto é {a} menos {b}?",
                f"Calcule {a} - {b}.",
                f"Se eu subtrair {b} de {a}, quanto resta?",
            ])
            r = (
                f"Subtraindo {b} de {a}, realizamos a operação {a} - {b} = {ans}. "
                f"Resposta: {ans}."
            )
        elif op == "multiplicacao":
            a = random.randint(2, 50)
            b = random.randint(2, 20)
            ans = a * b
            q = random.choice([
                f"Quanto é {a} multiplicado por {b}?",
                f"Calcule {a} x {b}.",
                f"Qual o produto de {a} e {b}?",
            ])
            r = (
                f"{a} multiplicado por {b} equivale a somar {a} repetido {b} vezes, "
                f"totalizando {ans}. Resposta: {ans}."
            )
        elif op == "divisao":
            b = random.randint(2, 12)
            ans = random.randint(1, 30)
            a = b * ans
            q = random.choice([
                f"Divida {a} por {b}.",
                f"Quanto é {a} dividido por {b}?",
                f"Qual o resultado de {a} / {b}?",
            ])
            r = (
                f"Para dividir {a} por {b}, procuramos qual número vezes {b} dá {a}. "
                f"Como {ans} x {b} = {a}, temos {a} / {b} = {ans}. Resposta: {ans}."
            )

        prompt = f"<|user|>\n{q}\n<|assistant|>\n"
        resp = f"{r}\n<|endoftext|>\n"
        pairs.append((prompt, resp))

    return pairs

def generate_logic_sft_pairs(count: int = 6000):
    """Gera pares (prompt, response) de raciocínio lógico e problemas de contagem."""
    pairs = []
    itens = ["maçãs", "laranjas", "livros", "canetas", "moedas", "figurinhas", "balas"]

    for _ in range(count):
        item = random.choice(itens)
        n1 = random.randint(10, 80)
        n2 = random.randint(2, n1 - 1)
        sobra = n1 - n2

        nomes = [("Lucas", "seu irmão"), ("Maria", "sua amiga"), ("Pedro", "sua colega"), ("Ana", "seu primo")]
        nome, amigo = random.choice(nomes)

        q = f"{nome} tinha {n1} {item} e deu {n2} para {amigo}. Com quantas {item} {nome} ficou?"
        r = (
            f"Vamos resolver passo a passo:\n"
            f"1. Quantidade inicial: {n1} {item}.\n"
            f"2. Quantidade doada: {n2} {item}.\n"
            f"3. Subtração: {n1} - {n2} = {sobra}.\n"
            f"Resposta: {nome} ficou com {sobra} {item}."
        )

        prompt = f"<|user|>\n{q}\n<|assistant|>\n"
        resp = f"{r}\n<|endoftext|>\n"
        pairs.append((prompt, resp))

    return pairs

def build_multitask_sft(out_path: str = "data/sft_multitask_test.bin", target_tokens: int = 1_500_000):
    enc = get_tokenizer()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=" * 75)
    print("  GERANDO DATASET SFT MULTI-TAREFA COM MÁSCARA DE LOSS (LOSS MASKING)")
    print("=" * 75)

    # 1. Diálogos de Conversação Geral (Alpaca-PT)
    print("\n[1/3] Carregando diálogos conversacionais em português (Alpaca-PT)...")
    chat_pairs = []
    try:
        ds = load_dataset("dominguesm/alpaca-data-pt-br", split="train", streaming=True)
        for row in ds:
            instruction = row.get("instruction", "").strip()
            inp = row.get("input", "").strip()
            output = row.get("output", "").strip()

            if not instruction or not output:
                continue

            user_msg = f"{instruction} {inp}".strip()
            prompt = f"<|user|>\n{user_msg}\n<|assistant|>\n"
            resp = f"{output}\n<|endoftext|>\n"
            chat_pairs.append((prompt, resp))

            if len(chat_pairs) >= 35000:
                break
        print(f" -> {len(chat_pairs):,} diálogos de conversação carregados.")
    except Exception as e:
        print(f" -> Aviso ao carregar Alpaca-PT: {e}.")

    # 2. Diálogos de Matemática e Lógica
    print("\n[2/3] Gerando diálogos de Matemática CoT e Lógica...")
    math_pairs = generate_math_sft_pairs(count=12000)
    logic_pairs = generate_logic_sft_pairs(count=6000)
    print(f" -> {len(math_pairs):,} diálogos de matemática CoT gerados.")
    print(f" -> {len(logic_pairs):,} diálogos de lógica gerados.")

    # 3. Mistura e aplicação da Máscara de Loss
    print("\n[3/3] Aplicando Máscara de Loss (IGNORE_INDEX = 65535 nos prompts)...")
    all_pairs = chat_pairs + math_pairs + logic_pairs
    random.seed(42)
    random.shuffle(all_pairs)

    token_flat = []
    total_tokens = 0
    masked_tokens = 0
    active_tokens = 0

    idx = 0
    while total_tokens < target_tokens:
        if idx >= len(all_pairs):
            random.shuffle(all_pairs)
            idx = 0

        prompt, resp = all_pairs[idx]
        idx += 1

        p_toks = enc.encode(prompt, allowed_special={"<|endoftext|>"})
        r_toks = enc.encode(resp, allowed_special={"<|endoftext|>"})

        seq = p_toks + r_toks
        # Cria os pares (input, target)
        for i in range(len(seq) - 1):
            inp = seq[i]
            # Se for token do prompt, mascara com 65535
            if i < len(p_toks) - 1:
                tgt = 65535
                masked_tokens += 1
            else:
                tgt = seq[i + 1]
                active_tokens += 1

            token_flat.append(inp)
            token_flat.append(tgt)
            total_tokens += 1

            if total_tokens >= target_tokens:
                break

    arr = np.array(token_flat, dtype=np.uint16)
    arr.tofile(out_path)

    mb_size = os.path.getsize(out_path) / (1024 * 1024)
    pct_active = (active_tokens / total_tokens) * 100.0
    pct_masked = (masked_tokens / total_tokens) * 100.0

    print("\n" + "=" * 75)
    print(f" [SUCESSO] Dataset SFT Multi-Tarefa com Máscara gerado em: {out_path}")
    print(f" -> Total de tokens: {total_tokens:,} ({mb_size:.2f} MB)")
    print(f" -> Tokens do Assistente (Treinados com gradiente ativo): {active_tokens:,} ({pct_active:.1f}%)")
    print(f" -> Tokens do Prompt/User (Mascarados com 65535 - Zero Erro):  {masked_tokens:,} ({pct_masked:.1f}%)")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera dataset SFT multi-tarefa com máscara de loss")
    parser.add_argument("--out", type=str, default="data/sft_multitask_test.bin")
    parser.add_argument("--tokens", type=int, default=1_500_000)
    args = parser.parse_args()

    build_multitask_sft(args.out, args.tokens)
