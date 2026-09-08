#!/usr/bin/env python3
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from chat import StackedRNNGenerator

def test_math():
    ckpt = "checkpoints/sore_lm_150m_pretrain.bin"
    bot = StackedRNNGenerator(ckpt)

    tests = [
        ("Multiplicação CoT", "Problema: Quanto é 6 multiplicado por 7?\nRaciocínio: 6 vezes 7 equivale a somar 6 repetido 7 vezes, totalizando"),
        ("Multiplicação Direta", "Problema: Quanto é 6 multiplicado por 7?\nRaciocínio:"),
        ("Soma CoT", "Problema: Calcule 15 + 27.\nRaciocínio: Para somar 15 e 27, decompomos os valores:"),
        ("Soma Direta", "Problema: Calcule 15 + 27.\nRaciocínio:"),
        ("Subtração CoT", "Problema: Quanto é 50 menos 20?\nRaciocínio: Subtraindo 20 de 50,"),
        ("Subtração Direta", "Problema: Quanto é 50 menos 20?\nRaciocínio:"),
        ("Equação Pura", "15 + 27 ="),
        ("Tabuada Pura", "6 x 7 =")
    ]

    print("=== TESTE DE RETENÇÃO MATEMÁTICA NO PASSO 37.500 ===")
    for name, prompt in tests:
        print(f"\n[{name}]")
        print(f"Prompt: {prompt.replace(chr(10), ' ')}")
        resp = bot.generate(prompt, max_new_tokens=30, temperature=0.1)
        print(f"Saída: {resp.strip()}")

if __name__ == "__main__":
    test_math()
