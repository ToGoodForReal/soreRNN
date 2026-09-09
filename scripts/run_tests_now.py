#!/usr/bin/env python3
"""
Bateria de Testes Rápidos e Avaliação Cognitiva do soreRNN-LM v2
Executa 10 testes representativos de Matemática, Lógica, Raciocínio e Conhecimento Geral.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from chat import StackedRNNGenerator

def run_tests():
    ckpt = "checkpoints/sore_lm_150m_pretrain.bin"
    if not os.path.exists(ckpt):
        print(f"Checkpoint não encontrado: {ckpt}")
        return

    print("=" * 70)
    print(f"  AVALIAÇÃO COGNITIVA DO soreRNN-LM v2 (GPU ACELERADA)")
    print(f"  Checkpoint: {ckpt}")
    print("=" * 70)

    t0 = time.time()
    bot = StackedRNNGenerator(ckpt)
    print(f"Modelo carregado em {time.time()-t0:.2f}s\n")

    test_cases = [
        ("Aritmética Básica (Soma)", "Calcule 15 + 27."),
        ("Aritmética Básica (Subtração)", "Quanto é 50 menos 20?"),
        ("Aritmética Básica (Multiplicação)", "Quanto é 6 multiplicado por 7?"),
        ("Aritmética Básica (Divisão)", "Divida 40 por 8."),
        ("Aritmética 3 Dígitos (Soma)", "Calcule 150 + 250."),
        ("Raciocínio Lógico (Dedução)", "Problema: Se Lucas tinha 20 figurinhas e deu 5 para Pedro, com quantas Lucas ficou?"),
        ("Raciocínio Lógico (Ordem)", "Problema: Se João é mais alto que Pedro e Pedro é mais alto que Carlos, quem é o mais alto?"),
        ("Conhecimento Geral (Geografia)", "Pergunta: Qual é a capital do Brasil?"),
        ("Conhecimento Geral (Ciência)", "Pergunta: Qual é a fórmula química da água?"),
        ("Português (Gramática)", "Pergunta: Qual é o antônimo de 'alto'?")
    ]

    results = []
    print("Iniciando execução dos 10 testes cognitivos...\n")

    for idx, (category, prompt_text) in enumerate(test_cases, 1):
        if prompt_text.startswith("Problema:") or prompt_text.startswith("Pergunta:"):
            formatted = prompt_text + "\nRaciocínio:"
        else:
            formatted = f"Problema: {prompt_text}\nRaciocínio:"

        print(f"[{idx}/10] Categoria: {category}")
        print(f"  Prompt de Entrada: \"{prompt_text}\"")
        
        # Gera com temperatura 0.2 para precisão
        output = bot.generate(formatted, max_new_tokens=50, temperature=0.2, repetition_penalty=1.15).strip()
        print(f"  Saída soreRNN:\n  -> {output}")
        print("-" * 70)

        results.append({
            "idx": idx,
            "category": category,
            "prompt": prompt_text,
            "output": output
        })

    out_file = "logs/test_evaluation_report.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("RELATÓRIO DE AVALIAÇÃO COGNITIVA soreRNN-LM v2\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write(f"[{r['idx']}] {r['category']}\n")
            f.write(f"Prompt: {r['prompt']}\n")
            f.write(f"Saída: {r['output']}\n\n")
    print(f"\n[OK] Relatório completo gravado em: {out_file}")

if __name__ == "__main__":
    run_tests()
