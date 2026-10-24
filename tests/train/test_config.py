import pytest

from sore.train.config import parse_overrides


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
