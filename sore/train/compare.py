"""Portões de qualidade: perda final de uma corrida e a regra de comparação com a referência.

Cada alavanca de velocidade roda uma corrida curta (`configs/s150m_quality.yaml`) e a perda final
dela é comparada com a de duas referências (seeds 0 e 1, tudo desligado):
- exata (não deveria mudar a matemática): |gap| <= max(ruído entre as duas seeds, 0,5%);
- com perda (aproximação aceita em troca de velocidade): gap <= 1%, só para cima.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

EXACT_FLOOR = 0.005  # piso do limiar exato: abaixo disso nem a diferença entre seeds é distinguível
LOSSY_LIMIT = 0.01


@dataclass(frozen=True)
class GateResult:
    passed: bool
    gap_rel: float  # (cand - ref) / ref
    threshold_rel: float


def final_loss(metrics_path: Path, key: str = "lm_loss", last_frac: float = 0.1) -> float:
    """Média de `key` nos últimos `ceil(last_frac × n)` valores não nulos do `metrics.jsonl`.

    Passos pulados (não finitos) gravam `null`; eles não contam nem no `n` nem na média. A média
    das últimas linhas, e não o último passo, tira o ruído de um único batch.
    """
    values = []
    for line in Path(metrics_path).read_text().splitlines():
        if line.strip():
            value = json.loads(line).get(key)
            if value is not None:
                values.append(float(value))
    if not values:
        raise ValueError(f"{metrics_path}: nenhum valor não nulo de {key!r}")
    k = max(1, math.ceil(last_frac * len(values)))
    return sum(values[-k:]) / k


def gate(ref: float, cand: float, *, mode: Literal["exact", "lossy"], noise: float | None = None) -> GateResult:
    """Aplica a regra de qualidade; `noise` é o ruído relativo entre as duas referências."""
    if ref <= 0:
        raise ValueError(f"perda de referência deve ser > 0, recebeu {ref}")
    gap_rel = (cand - ref) / ref
    if mode == "exact":
        threshold = max(noise or 0.0, EXACT_FLOOR)
        return GateResult(abs(gap_rel) <= threshold, gap_rel, threshold)
    if mode == "lossy":
        return GateResult(gap_rel <= LOSSY_LIMIT, gap_rel, LOSSY_LIMIT)
    raise ValueError(f"mode deve ser 'exact' ou 'lossy', recebeu {mode!r}")
