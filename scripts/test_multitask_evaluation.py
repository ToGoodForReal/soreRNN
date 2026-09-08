#!/usr/bin/env python3
"""
Script de Avaliação Rápida para soreRNN-LM v2:
Compara as capacidades de Conversação, Matemática CoT, Lógica e Conhecimento Geral.
Executa em CPU/NumPy sem consumir a VRAM da GPU utilizada pelo treino principal.
"""

import os
import sys
import argparse
import numpy as np
import tiktoken
from chat import StackedRNNGenerator

def generate_completion(gen: StackedRNNGenerator, prompt: str, max_new_tokens: int = 35, temperature: float = 0.5):
    tokens = gen.enc.encode(prompt, allowed_special={"<|user|>", "<|assistant|>", "<|endoftext|>"})
    h_states = [np.zeros(gen.d_model, dtype=np.float32) for _ in range(gen.num_layers)]
    conv_buffers = [np.zeros((3, gen.d_model), dtype=np.float32) for _ in range(gen.num_layers)]

    logits = None
    for tok in tokens:
        logits, h_states, conv_buffers = gen.step(tok, h_states, conv_buffers)

    generated_tokens = []
    current_tok = tokens[-1] if tokens else 0
    all_tokens = list(tokens)

    for _ in range(max_new_tokens):
        if logits is None:
            logits, h_states, conv_buffers = gen.step(current_tok, h_states, conv_buffers)

        next_tok = gen.sample_token(
            logits,
            temperature=temperature,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.15,
            context_tokens=all_tokens
        )
        if next_tok == 50256: # <|endoftext|>
            break

        generated_tokens.append(next_tok)
        all_tokens.append(next_tok)
        current_tok = next_tok
        logits = None

        word = gen.enc.decode([next_tok])
        if "<|user|>" in word or "<|endoftext|>" in word or "\n\n\n" in word:
            break

    return gen.enc.decode(generated_tokens).strip()

def evaluate_checkpoint(ckpt_path: str, max_new_tokens: int = 35):
    print("=" * 70)
    print(f"  AVALIAÇÃO MULTI-TAREFA soreRNN-LM: {os.path.basename(ckpt_path)}")
    print("=" * 70)

    gen = StackedRNNGenerator(ckpt_path)

    test_cases = [
        {
            "category": "Saudação & Conversação",
            "prompt": "<|user|>\nOlá! Quem é você e como pode me ajudar?\n<|assistant|>\n",
            "expected_keywords": ["sou", "ajudar", "assistente", "olá", "posso"],
        },
        {
            "category": "Conversação Geral & Dicas",
            "prompt": "<|user|>\nDê uma dica simples para ter uma vida saudável.\n<|assistant|>\n",
            "expected_keywords": ["alimentação", "água", "exercício", "sono", "saúde", "dieta", "corpo"],
        },
        {
            "category": "Aritmética (Multiplicação CoT)",
            "prompt": "<|user|>\nQuanto é 6 multiplicado por 7?\n<|assistant|>\n",
            "expected_keywords": ["42", "6", "7"],
        },
        {
            "category": "Aritmética (Soma CoT)",
            "prompt": "<|user|>\nCalcule 15 + 27.\n<|assistant|>\n",
            "expected_keywords": ["42", "15", "27"],
        },
        {
            "category": "Raciocínio Lógico (Problema de Contagem)",
            "prompt": "<|user|>\nLucas tinha 20 figurinhas e deu 5 para seu amigo. Com quantas figurinhas Lucas ficou?\n<|assistant|>\n",
            "expected_keywords": ["15", "Lucas", "ficou"],
        },
        {
            "category": "Conhecimento Geral (Geografia)",
            "prompt": "<|user|>\nQual é a capital do Brasil?\n<|assistant|>\n",
            "expected_keywords": ["Brasília", "capital"],
        },
    ]

    total_score = 0.0

    for idx, tc in enumerate(test_cases, 1):
        prompt = tc["prompt"]
        cat = tc["category"]
        print(f"\n[{idx}/6] Categoria: {cat}")
        print(f"Pergunta:")
        clean_p = prompt.replace("<|user|>\n", "").replace("\n<|assistant|>\n", "").strip()
        print(f"  \"{clean_p}\"")

        resp_text = generate_completion(gen, prompt, max_new_tokens=max_new_tokens)
        print(f"Resposta:")
        print(f"  \"{resp_text}\"")

        # Avaliação de coerência e keywords
        hits = sum(1 for kw in tc["expected_keywords"] if kw.lower() in resp_text.lower())
        cat_score = min(1.0, hits / max(1, len(tc["expected_keywords"]) * 0.4))
        if len(resp_text) > 10 and not resp_text.startswith("!"):
            cat_score = max(cat_score, 0.4)
        total_score += cat_score
        print(f"  -> Aderência ao Tópico: {cat_score * 10.0:.1f} / 10.0")

    final_score = (total_score / len(test_cases)) * 10.0
    print("\n" + "=" * 70)
    print(f" NOTA GERAL MULTI-TAREFA: {final_score:.2f} / 10.0")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sore_lm_150m_test_base.bin")
    parser.add_argument("--tokens", type=int, default=35)
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, args.tokens)
