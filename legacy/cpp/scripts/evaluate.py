#!/usr/bin/env python3
"""
soreRNN-LM — Avaliação Multi-Tarefa Unificada (Conversação, Matemática, Lógica, Geral)
Pontos de entrada únicos de avaliação de modelo (era disperso em vários scripts).

Uso:
    python3 scripts/evaluate.py --checkpoint checkpoints/sore_lm_150m_sft.bin [--tokens 40] [--mode pt|gpt2]
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chat import StackedRNNGenerator

TASKS = [
    {"cat": "Conversação",        "prompt": "<|user|>\nOlá! Quem é você e como pode me ajudar?\n<|assistant|>\n",
     "keys": ["sou", "ajudar", "assistente", "posso", "olá"]},
    {"cat": "Saúde / Dica",       "prompt": "<|user|>\nDê uma dica simples para ter uma vida saudável.\n<|assistant|>\n",
     "keys": ["água", "exercício", "sono", "aliment", "saúde", "dieta"]},
    {"cat": "Matemática (×)",     "prompt": "<|user|>\nQuanto é 6 multiplicado por 7?\n<|assistant|>\n",
     "keys": ["42", "6", "7"]},
    {"cat": "Matemática (+)",     "prompt": "<|user|>\nCalcule 15 + 27.\n<|assistant|>\n",
     "keys": ["42", "15", "27"]},
    {"cat": "Lógica (contagem)",  "prompt": "<|user|>\nLucas tinha 20 figurinhas e deu 5 para um amigo. Com quantas ficou?\n<|assistant|>\n",
     "keys": ["15", "lucas", "ficou"]},
    {"cat": "Conhecimento geral", "prompt": "<|user|>\nQual é a capital do Brasil?\n<|assistant|>\n",
     "keys": ["brasília", "capital"]},
]

def score(resp, keys):
    r = resp.lower()
    hits = sum(1 for k in keys if k in r)
    s = min(1.0, hits / max(1, len(keys) * 0.4))
    if len(resp) > 10 and not resp.startswith("!"):
        s = max(s, 0.4)
    return s

def main():
    ap = argparse.ArgumentParser(description="Avaliação multi-tarefa do soreRNN-LM")
    ap.add_argument("--checkpoint", default="checkpoints/sore_lm_150m_pretrain.bin")
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--mode", default="pt", choices=["pt", "gpt2", "llama"])
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"[Erro] Checkpoint não encontrado: {args.checkpoint}")

    print("=" * 66)
    print(f"  AVALIAÇÃO soreRNN-LM: {os.path.basename(args.checkpoint)}  (tok={args.mode})")
    print("=" * 66)
    gen = StackedRNNGenerator(args.checkpoint, tokenizer_name=args.mode)

    total = 0.0
    for i, t in enumerate(TASKS, 1):
        out = gen.generate(t["prompt"], max_new_tokens=args.tokens,
                           temperature=0.3, top_k=40, top_p=0.9,
                           repetition_penalty=1.15).strip()
        s = score(out, t["keys"]); total += s
        clean = t["prompt"].replace("<|user|>\n", "").split("\n<|assistant|>")[0].strip()
        print(f"\n[{i}/{len(TASKS)}] {t['cat']}  (aderência {s*10:.1f}/10)")
        print(f"  Q: {clean}")
        print(f"  R: {out[:200]}")

    print("\n" + "=" * 66)
    print(f"  NOTA GERAL: {(total/len(TASKS))*10:.2f} / 10.0")
    print("=" * 66)

if __name__ == "__main__":
    main()
