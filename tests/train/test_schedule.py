"""Testes do multiplicador de LR warmup-stable-decay (WSD) e da rampa linear (λ do MTP)."""
import math

import pytest

from sore.train.schedule import linear_schedule, wsd_multiplier


def test_wsd_points():
    # total=100, decay_frac=0.2 -> decay começa no passo 80
    assert wsd_multiplier(0, 10, 100, 0.2) == 0.0
    assert wsd_multiplier(10, 10, 100, 0.2) == 1.0
    assert wsd_multiplier(50, 10, 100, 0.2) == 1.0
    assert wsd_multiplier(100, 10, 100, 0.2) == 0.0
    assert wsd_multiplier(90, 10, 100, 0.2) == pytest.approx(1 - math.sqrt(0.5))


def test_wsd_warmup_is_linear():
    assert wsd_multiplier(5, 10, 100, 0.2) == pytest.approx(0.5)
    assert wsd_multiplier(1, 10, 100, 0.2) == pytest.approx(0.1)


def test_wsd_plateau_edges():
    assert wsd_multiplier(79, 10, 100, 0.2) == 1.0
    assert wsd_multiplier(80, 10, 100, 0.2) == 1.0  # progresso 0 -> 1 - sqrt(0) = 1


def test_wsd_decay_monotone_and_bounded():
    values = [wsd_multiplier(s, 10, 100, 0.2) for s in range(80, 101)]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert values[0] == 1.0 and values[-1] == 0.0
    # o último passo de treino (99) ainda tem LR positivo
    assert 0.0 < wsd_multiplier(99, 10, 100, 0.2) < 0.2


def test_wsd_past_total_is_zero():
    assert wsd_multiplier(101, 10, 100, 0.2) == 0.0
    assert wsd_multiplier(10_000, 10, 100, 0.2) == 0.0


def test_wsd_no_decay_phase():
    # decay_frac=0: plateau até o fim; só total zera
    assert wsd_multiplier(99, 10, 100, 0.0) == 1.0
    assert wsd_multiplier(100, 10, 100, 0.0) == 0.0


def test_wsd_no_warmup():
    assert wsd_multiplier(0, 0, 100, 0.2) == 1.0


def test_wsd_all_decay():
    # decay_frac=1 com warmup=0: decai desde o passo 0
    assert wsd_multiplier(0, 0, 100, 1.0) == 1.0
    assert wsd_multiplier(25, 0, 100, 1.0) == pytest.approx(0.5)  # progresso 0.25


@pytest.mark.parametrize(
    "args",
    [
        (0, -1, 100, 0.2),  # warmup < 0
        (0, 10, 0, 0.2),  # total <= 0
        (0, 10, 100, -0.1),  # decay_frac < 0
        (0, 10, 100, 1.1),  # decay_frac > 1
        (0, 90, 100, 0.2),  # warmup > decay_start (80)
    ],
)
def test_wsd_rejects_invalid_args(args):
    with pytest.raises(ValueError):
        wsd_multiplier(*args)


def test_linear_schedule_endpoints_and_mid():
    # λ do MTP (R23): 0.3 no passo 0 e 0.1 no último passo (total - 1), linear no meio
    assert linear_schedule(0, 101, 0.3, 0.1) == 0.3
    assert linear_schedule(50, 101, 0.3, 0.1) == pytest.approx(0.2)
    assert linear_schedule(100, 101, 0.3, 0.1) == pytest.approx(0.1)
    assert linear_schedule(150, 101, 0.3, 0.1) == pytest.approx(0.1)  # depois do fim fica no valor final
    assert linear_schedule(0, 1, 0.3, 0.1) == 0.3  # um passo só
    with pytest.raises(ValueError):
        linear_schedule(0, 0, 0.3, 0.1)
