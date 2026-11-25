"""Mistura do pré-treino e mistura de calibração da poda do tokenizer (`sore.train.sources`)."""
import pytest

from sore.train.sources import CODE_SOURCES, MIXTURE, SOURCES, calibration_mixture


def test_mixture_sums_to_one_and_names_known_sources():
    assert sum(MIXTURE.values()) == pytest.approx(1.0)
    assert set(MIXTURE) <= set(SOURCES)
    assert set(CODE_SOURCES) <= set(MIXTURE)
    assert sum(MIXTURE[n] for n in CODE_SOURCES) == pytest.approx(0.10)  # spec: 10% de código no treino


def test_calibration_mixture_sets_code_share_and_keeps_proportions():
    cal = calibration_mixture(0.25)
    assert set(cal) == set(MIXTURE)
    assert sum(cal.values()) == pytest.approx(1.0)
    assert sum(cal[n] for n in CODE_SOURCES) == pytest.approx(0.25)
    # dentro de cada grupo, as proporções do treino continuam as mesmas
    assert cal["github_code"] / cal["opc_annealing"] == pytest.approx(MIXTURE["github_code"] / MIXTURE["opc_annealing"])
    assert cal["fineweb2_pt"] / cal["fineweb_edu"] == pytest.approx(MIXTURE["fineweb2_pt"] / MIXTURE["fineweb_edu"])


def test_calibration_mixture_at_training_share_is_the_mixture():
    cal = calibration_mixture(0.10)
    assert cal == pytest.approx(MIXTURE)


def test_calibration_mixture_without_code():
    cal = calibration_mixture(0.0)
    assert all(n not in cal for n in CODE_SOURCES)
    assert sum(cal.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("frac", [-0.1, 1.0, 1.5])
def test_calibration_mixture_rejects_out_of_range(frac):
    with pytest.raises(ValueError, match="code_frac"):
        calibration_mixture(frac)


def test_calibration_mixture_needs_code_sources():
    with pytest.raises(ValueError, match="código"):
        calibration_mixture(0.2, {"fineweb2_pt": 1.0})
