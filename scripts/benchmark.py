#!/usr/bin/env python3
"""
Suíte de Benchmark Automatizada para o soreRNN-LM (152M Parâmetros).
Avalia:
1. Matemática Básica (Aritmética de 2 e 3 dígitos: Adição, Subtração, Multiplicação)
2. Raciocínio Lógico e Dedução Direta
3. Conhecimentos Gerais (Ciência, Geografia, História)
4. Fluência e Comunicação em Português

Calcula o Índice de Inteligência Geral (0 a 10 Pontos).
"""

import os
import sys
import re
import json
import argparse
import numpy as np

# Importa o gerador do soreRNN
sys.path.insert(0, os.path.dirname(__file__))
from chat import StackedRNNGenerator

BENCHMARK_TASKS = {
    "aritmetica": [
        {"prompt": "Quanto é 15 + 27?\nResposta:", "expected": ["42"]},
        {"prompt": "Quanto é 7 x 8?\nResposta:", "expected": ["56"]},
        {"prompt": "Quanto é 100 - 45?\nResposta:", "expected": ["55"]},
        {"prompt": "Quanto é 12 x 4?\nResposta:", "expected": ["48"]},
        {"prompt": "Quanto é 50 + 75?\nResposta:", "expected": ["125"]},
        {"prompt": "Quanto é 9 x 6?\nResposta:", "expected": ["54"]},
        {"prompt": "Quanto é 80 / 4?\nResposta:", "expected": ["20"]},
        {"prompt": "Quanto é 33 + 67?\nResposta:", "expected": ["100"]},
        {"prompt": "Quanto é 8 x 8?\nResposta:", "expected": ["64"]},
        {"prompt": "Quanto é 200 - 80?\nResposta:", "expected": ["120"]},
    ],
    "raciocinio": [
        {
            "prompt": "Se João é mais alto que Pedro e Pedro é mais alto que Carlos, quem é o mais alto?\nResposta:",
            "expected": ["João", "joao"]
        },
        {
            "prompt": "O céu durante o dia sem nuvens é de que cor?\nResposta:",
            "expected": ["azul"]
        },
        {
            "prompt": "Se um carro tem 4 rodas, quantos pneus têm 2 carros?\nResposta:",
            "expected": ["8", "oito"]
        },
        {
            "prompt": "Se todos os gatos miam e Mingau é um gato, o que Mingau faz?\nResposta:",
            "expected": ["mia", "miar"]
        },
        {
            "prompt": "O que vem depois do número 9?\nResposta:",
            "expected": ["10", "dez"]
        },
    ],
    "conhecimento_geral": [
        {"prompt": "Qual é a capital do Brasil?\nResposta:", "expected": ["Brasília", "brasilia"]},
        {"prompt": "Qual é o maior planeta do sistema solar?\nResposta:", "expected": ["Júpiter", "jupiter"]},
        {"prompt": "Qual é a fórmula química da água?\nResposta:", "expected": ["H2O", "h2o"]},
        {"prompt": "Em qual continente fica o Brasil?\nResposta:", "expected": ["América do Sul", "america do sul"]},
        {"prompt": "Qual órgão humano bombeia sangue para o corpo?\nResposta:", "expected": ["coração", "coracao"]},
    ],
    "portugues_fluencia": [
        {
            "prompt": "Complete a frase: 'Ontem nós ______ ao cinema.' (fomos / fomos-nos)\nResposta:",
            "expected": ["fomos"]
        },
        {
            "prompt": "Qual é o antônimo de 'alto'?\nResposta:",
            "expected": ["baixo"]
        },
        {
            "prompt": "Qual é o plural da palavra 'cão'?\nResposta:",
            "expected": ["cães", "caes"]
        },
        {
            "prompt": "Diga 'olá' de forma educada em português.\nResposta:",
            "expected": ["olá", "ola", "bom dia", "boa tarde"]
        },
    ]
}

def evaluate_model(checkpoint_path: str, max_tokens: int = 25, temp: float = 0.2):
    print("=" * 65)
    print(f"  AVALIAÇÃO DE BENCHMARK DO soreRNN-LM (152M)")
    print(f"  Checkpoint: {checkpoint_path}")
    print("=" * 65)

    bot = StackedRNNGenerator(checkpoint_path)
    results = {}
    total_tests = 0
    total_passed = 0

    is_pretrain = "pretrain" in checkpoint_path
    for category, items in BENCHMARK_TASKS.items():
        cat_passed = 0
        print(f"\n--- Testando Categoria: {category.upper()} ({len(items)} questões) ---")

        for i, item in enumerate(items, 1):
            raw_prompt = item['prompt'].strip()
            if is_pretrain:
                if category == "aritmetica":
                    clean_p = raw_prompt.replace("\nResposta:", "")
                    prompt = f"Problema: {clean_p}\nRaciocínio:"
                elif category == "raciocinio":
                    clean_p = raw_prompt.replace("\nResposta:", "")
                    prompt = f"Problema: {clean_p}\nRaciocínio:"
                else:
                    clean_p = raw_prompt.replace("\nResposta:", "")
                    prompt = f"Pergunta: {clean_p}\nResposta:"
            else:
                prompt = f"<|user|>\n{raw_prompt}\n<|assistant|>\n"

            # Geração com baixa temperatura para respostas factuais determinísticas
            output = bot.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temp,
                repetition_penalty=1.1
            ).strip()

            passed = False
            for exp in item["expected"]:
                # Normalização de busca case-insensitive e pontuação
                clean_exp = exp.lower().strip()
                clean_out = output.lower()
                if clean_exp in clean_out:
                    passed = True
                    break

            status = "ACERTO" if passed else "ERRO"
            if passed:
                cat_passed += 1
                total_passed += 1
            total_tests += 1

            print(f"  [{status}] Q{i}: '{item['prompt'].replace(chr(10), ' ')}'")
            print(f"         Esperado: {item['expected']} | Gerado: '{output[:40]}...'")

        cat_acc = (cat_passed / len(items)) * 100.0
        results[category] = {
            "passed": cat_passed,
            "total": len(items),
            "accuracy": cat_acc
        }
        print(f" -> Desempenho {category}: {cat_passed}/{len(items)} ({cat_acc:.1f}%)")

    overall_acc = (total_passed / total_tests) * 100.0
    intelligence_score = overall_acc / 10.0 # Escala de 0 a 10 pontos

    print("\n" + "=" * 65)
    print("  RELATÓRIO FINAL DE INTELIGÊNCIA GERAL (soreRNN-LM)")
    print("=" * 65)
    for cat, data in results.items():
        print(f" - {cat.ljust(20)}: {data['passed']:2d}/{data['total']:2d} ({data['accuracy']:.1f}%)")
    print("-" * 65)
    print(f" TOTAL DE ACERTOS      : {total_passed}/{total_tests} ({overall_acc:.1f}%)")
    print(f" PONTOS DE INTELIGÊNCIA: {intelligence_score:.2f} / 10.0 PONTOS")
    print("=" * 65)

    report_path = "logs/benchmark_report.json"
    os.makedirs("logs", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": checkpoint_path,
            "intelligence_score": intelligence_score,
            "overall_accuracy": overall_acc,
            "total_passed": total_passed,
            "total_tests": total_tests,
            "categories": results
        }, f, indent=2, ensure_ascii=False)
    print(f" -> Relatório gravado em: {report_path}")

    return intelligence_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark de Inteligência do soreRNN-LM")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sore_lm_150m_final.bin")
    parser.add_argument("--temp", type=float, default=0.2)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        if os.path.exists("checkpoints/sore_lm_150m_pretrain.bin"):
            args.checkpoint = "checkpoints/sore_lm_150m_pretrain.bin"

    evaluate_model(args.checkpoint, temp=args.temp)
