import pytest

from sore.train.config import Stage, TrainConfig, parse_overrides


def test_stage_grad_accum_validation():
    # stages válidos com grad_accum por estágio e tokens_per_step == 1024
    stages = [
        Stage(until_tokens=10_000, seq_len=128, micro_batch=4, grad_accum=2),
        Stage(until_tokens=20_000, seq_len=256, micro_batch=1, grad_accum=4),
    ]
    cfg = TrainConfig(
        model_preset="tiny",
        data_paths=["data/*.bin"],
        eos_id=0,
        grad_accum=2,
        stages=stages,
        total_tokens=20_000,
        ckpt_every_tokens=5_000,
        out_dir="runs/test",
    )
    assert cfg.tokens_per_step == 1024
    assert cfg.stage_grad_accum(stages[0]) == 2
    assert cfg.stage_grad_accum(stages[1]) == 4

    # trocar o segundo para grad_accum: 3 levanta ValueError
    invalid_stages = [
        Stage(until_tokens=10_000, seq_len=128, micro_batch=4, grad_accum=2),
        Stage(until_tokens=20_000, seq_len=256, micro_batch=1, grad_accum=3),
    ]
    with pytest.raises(ValueError, match="grad_accum"):
        TrainConfig(
            model_preset="tiny",
            data_paths=["data/*.bin"],
            eos_id=0,
            grad_accum=2,
            stages=invalid_stages,
            total_tokens=20_000,
            ckpt_every_tokens=5_000,
            out_dir="runs/test",
        )



def test_parse_overrides():
    assert parse_overrides(["n_layers=4", "model_overrides.fused_ce=true", "lr_adamw=3e-3"]) == {
        "n_layers": 4,
        "model_overrides": {"fused_ce": True},
        "lr_adamw": 3e-3,
    }
    with pytest.raises(ValueError):
        parse_overrides(["sem_igual"])


def test_parse_overrides_values():
    out = parse_overrides(["gdn_backend=fla", "x=35e9", "y=nan", "z=inf", "w=null", "v=[1, 2]", "e="])
    assert out["gdn_backend"] == "fla"
    assert out["x"] == 35e9 and isinstance(out["x"], float)
    assert out["y"] == "nan" and out["z"] == "inf"  # "nan"/"inf" nunca viram float
    assert out["w"] is None
    assert out["v"] == [1, 2]
    assert out["e"] is None  # YAML: valor vazio é null


def test_parse_overrides_nested_keys_merge():
    got = parse_overrides(["model_overrides.a=1", "model_overrides.b.c=x"])
    assert got == {"model_overrides": {"a": 1, "b": {"c": "x"}}}
