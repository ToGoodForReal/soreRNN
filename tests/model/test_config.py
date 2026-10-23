"""Testes do ModelConfig e dos presets."""
import pytest

from sore.model.config import PRESETS, ModelConfig, get_preset


def test_b0_8_layer_types():
    types = get_preset("b0_8").layer_types()
    assert len(types) == 24
    assert types.count("gdn") == 18
    assert types.count("attn") == 6
    assert types[3] == types[23] == "attn"


def test_invalid_layers_raises():
    # 6 camadas não é múltiplo do padrão de 4
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=512, d_model=128, n_layers=6, ffn_dim=384)


def test_invalid_head_ratios_raise():
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=512, d_model=128, n_layers=4, ffn_dim=384, gdn_qk_heads=4, gdn_v_heads=6)
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=512, d_model=128, n_layers=4, ffn_dim=384, attn_heads=6, attn_kv_heads=4)
    with pytest.raises(ValueError, match="loss_chunk_size"):
        ModelConfig(vocab_size=512, d_model=128, n_layers=4, ffn_dim=384, loss_chunk_size=0)


def test_roundtrip():
    c = get_preset("tiny")
    assert ModelConfig.from_dict(c.to_dict()) == c


def test_roundtrip_through_json_keeps_layer_pattern_tuple():
    import json

    c = get_preset("s150m", layer_pattern=("gdn", "attn"))
    d = json.loads(json.dumps(c.to_dict()))  # JSON/YAML devolvem lista
    assert isinstance(d["layer_pattern"], list)
    back = ModelConfig.from_dict(d)
    assert back == c
    assert isinstance(back.layer_pattern, tuple)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("tiny", dict(vocab_size=512, d_model=128, n_layers=4, ffn_dim=384,
                      gdn_qk_heads=1, gdn_v_heads=2, gdn_head_dim=64,
                      attn_heads=2, attn_kv_heads=1, attn_head_dim=64,
                      gdn_backend="torch", attn_backend="sdpa")),
        ("s150m", dict(vocab_size=65536, d_model=768, n_layers=12, ffn_dim=2048,
                       gdn_qk_heads=6, gdn_v_heads=12, attn_heads=6, attn_kv_heads=2,
                       gdn_backend="fla", attn_backend="flex")),
        ("b0_8", dict(vocab_size=65536, d_model=1280, n_layers=24, ffn_dim=4352,
                      gdn_qk_heads=10, gdn_v_heads=20, attn_heads=10, attn_kv_heads=2,
                      gdn_backend="fla", attn_backend="flex")),
    ],
)
def test_preset_values(name, expected):
    c = get_preset(name)
    for field, value in expected.items():
        assert getattr(c, field) == value, field
    assert c.mtp_layers == 1


def test_get_preset_overrides_do_not_mutate_presets():
    before = PRESETS["tiny"].to_dict()
    c = get_preset("tiny", grad_checkpoint=True, residual="hyper")
    assert c.grad_checkpoint is True and c.residual == "hyper"
    assert c is not PRESETS["tiny"]
    assert PRESETS["tiny"].to_dict() == before
    assert get_preset("tiny") is not PRESETS["tiny"]


def test_get_preset_unknown_name_lists_available():
    with pytest.raises(KeyError, match="tiny"):
        get_preset("nope")
