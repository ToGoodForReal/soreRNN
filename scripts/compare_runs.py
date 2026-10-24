"""Compara a perda final de corridas candidatas com as referências (portão de qualidade).

Uso: python scripts/compare_runs.py --ref runs/quality/ref_s0 --ref2 runs/quality/ref_s1 \
         runs/quality/<cand> ... [--lossy NOME ...] [--key lm_loss] [--last-frac 0.1]

O ruído é |final(ref2) - final(ref)| / final(ref). Por padrão o candidato é "exato" (|gap| <=
max(ruído, 0,5%)); os nomes em `--lossy` usam a regra com perda (gap <= 1%, só para cima).
Imprime uma tabela markdown e sai com código 1 se algum candidato REPROVA.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sore.train.compare import final_loss, gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", type=Path, required=True, help="corrida de referência (seed 0)")
    parser.add_argument("--ref2", type=Path, required=True, help="segunda referência (seed 1), só para o ruído")
    parser.add_argument("candidates", type=Path, nargs="*", help="diretórios das corridas candidatas")
    parser.add_argument("--lossy", action="append", default=[], metavar="NOME",
                        help="nome (basename) de candidato que usa a regra com perda; repetível")
    parser.add_argument("--key", default="lm_loss")
    parser.add_argument("--last-frac", type=float, default=0.1)
    args = parser.parse_args(argv)

    def final(run: Path) -> float:
        return final_loss(run / "metrics.jsonl", args.key, args.last_frac)

    ref, ref2 = final(args.ref), final(args.ref2)
    noise = abs(ref2 - ref) / ref
    print(f"Ruído entre as referências ({args.ref.name} x {args.ref2.name}): {noise:.3%} "
          f"(piso exato 0,500%)\n")
    print(f"| corrida | {args.key} final | gap | limiar | resultado |")
    print("|---|---|---|---|---|")
    print(f"| {args.ref.name} | {ref:.4f} | - | - | referência |")
    print(f"| {args.ref2.name} | {ref2:.4f} | - | - | referência |")
    failed = 0
    for cand in args.candidates:
        mode = "lossy" if cand.name in args.lossy else "exact"
        value = final(cand)
        result = gate(ref, value, mode=mode, noise=noise)
        failed += not result.passed
        verdict = "PASSA" if result.passed else "REPROVA"
        print(f"| {cand.name} ({mode}) | {value:.4f} | {result.gap_rel:+.3%} | "
              f"{'±' if mode == 'exact' else '≤ '}{result.threshold_rel:.3%} | {verdict} |")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
