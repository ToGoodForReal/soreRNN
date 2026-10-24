import json

import pytest

from sore.train.compare import GateResult, final_loss, gate


def test_gate_exact_uses_noise_floor():
    assert gate(2.000, 2.008, mode="exact").passed  # 0,4% < piso 0,5%
    assert not gate(2.000, 2.020, mode="exact", noise=0.004).passed
    assert gate(2.000, 2.020, mode="exact", noise=0.012).passed


def test_gate_exact_is_two_sided():
    # exata: melhorar demais também é suspeito (a alavanca não deveria mudar a matemática)
    assert not gate(2.0, 1.97, mode="exact").passed


def test_gate_lossy_is_one_sided():
    assert gate(2.0, 2.019, mode="lossy").passed and gate(2.0, 1.9, mode="lossy").passed
    assert not gate(2.0, 2.03, mode="lossy").passed


def test_gate_result_fields():
    r = gate(2.0, 2.02, mode="exact", noise=0.02)
    assert isinstance(r, GateResult)
    assert r.gap_rel == pytest.approx(0.01) and r.threshold_rel == pytest.approx(0.02)
    assert gate(2.0, 2.0, mode="lossy").threshold_rel == pytest.approx(0.01)


def test_gate_rejects_bad_input():
    with pytest.raises(ValueError):
        gate(2.0, 2.0, mode="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        gate(0.0, 1.0, mode="exact")


def _write_metrics(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_final_loss_last_fraction(tmp_path):
    # 20 linhas, algumas com null (passo pulado): valores válidos = 16 -> k = ceil(0,1 * 16) = 2
    rows = [{"step": i, "lm_loss": None if i in (3, 7, 12, 19) else float(i)} for i in range(1, 21)]
    path = tmp_path / "metrics.jsonl"
    _write_metrics(path, rows)
    valid = [r["lm_loss"] for r in rows if r["lm_loss"] is not None]
    assert valid[-2:] == [18.0, 20.0]  # a linha 19 é null: as 2 últimas válidas são 18 e 20
    assert final_loss(path) == pytest.approx(19.0)


def test_final_loss_key_and_rounding_up(tmp_path):
    rows = [{"step": i, "lm_loss": 1.0, "loss": float(i)} for i in range(1, 12)]  # 11 valores
    path = tmp_path / "metrics.jsonl"
    _write_metrics(path, rows)
    assert final_loss(path, key="loss") == pytest.approx((10.0 + 11.0) / 2)  # ceil(0,1 * 11) = 2
    assert final_loss(path, key="loss", last_frac=1.0) == pytest.approx(6.0)


def test_final_loss_without_values_raises(tmp_path):
    path = tmp_path / "metrics.jsonl"
    _write_metrics(path, [{"step": 1, "lm_loss": None}])
    with pytest.raises(ValueError, match="lm_loss"):
        final_loss(path)


def test_compare_runs_script_table_and_exit_code(tmp_path, capsys):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "compare_runs", Path(__file__).resolve().parents[2] / "scripts" / "compare_runs.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    def run(name, loss):
        d = tmp_path / name
        d.mkdir()
        _write_metrics(d / "metrics.jsonl", [{"step": 1, "lm_loss": loss}])
        return str(d)

    ref, ref2 = run("ref_s0", 2.000), run("ref_s1", 2.004)  # ruído 0,2% -> piso 0,5%
    ok, bad = run("ok", 2.006), run("bad", 2.030)
    args = ["--ref", ref, "--ref2", ref2]
    assert script.main(args + [ok]) == 0
    out = capsys.readouterr().out
    assert "| ok (exact) | 2.0060 |" in out and "PASSA" in out and "REPROVA" not in out
    assert script.main(args + [ok, bad]) == 1
    assert "REPROVA" in capsys.readouterr().out
    assert script.main(args + ["--lossy", "bad", bad]) == 1  # +1,5% > 1%
