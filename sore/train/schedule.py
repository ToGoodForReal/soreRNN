"""Schedules por passo: LR warmup-stable-decay (WSD) e rampa linear (λ do MTP)."""
from __future__ import annotations

import math


def wsd_multiplier(step: int, warmup: int, total: int, decay_frac: float) -> float:
    """Multiplicador do LR base no passo `step`, em [0, 1].

    - `step < warmup`: warmup linear `step / warmup` (0 no passo 0);
    - platô em 1.0 até `decay_start = total - decay_frac * total`;
    - nos últimos `decay_frac * total` passos: `1 - sqrt(progresso)`, com
      `progresso = (step - decay_start) / (total - decay_start)`;
    - `step >= total`: 0.0.
    """
    if warmup < 0:
        raise ValueError(f"warmup deve ser >= 0, recebeu {warmup}")
    if total <= 0:
        raise ValueError(f"total deve ser > 0, recebeu {total}")
    if not 0.0 <= decay_frac <= 1.0:
        raise ValueError(f"decay_frac deve estar em [0, 1], recebeu {decay_frac}")
    decay_start = total - decay_frac * total
    if warmup > decay_start:
        raise ValueError(f"warmup ({warmup}) não pode passar do início do decay ({decay_start})")

    if step >= total:
        return 0.0
    if step < warmup:
        mult = step / warmup
    elif step < decay_start:
        mult = 1.0
    else:
        mult = 1.0 - math.sqrt((step - decay_start) / (total - decay_start))
    return min(1.0, max(0.0, mult))


def linear_schedule(step: int, total: int, start: float, end: float) -> float:
    """`start` no passo 0 e `end` no último passo (`total - 1`), linear entre eles; `end` depois do fim.

    Usado no peso λ do MTP (ruling R23: 0.3 -> 0.1 ao longo do treino).
    """
    if total <= 0:
        raise ValueError(f"total deve ser > 0, recebeu {total}")
    frac = min(1.0, max(0.0, step / max(total - 1, 1)))
    return start + (end - start) * frac
