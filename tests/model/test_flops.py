"""Testes do modelo de FLOPs: limite 6N, monotonia em seq_len e contagem de matmuls contra o modelo."""
import pytest
import torch
from torch import nn

from sore.model import SoreLM, get_preset
from sore.model.flops import flops_per_token, matmul_params


def _meta_model(cfg) -> SoreLM:
    # device meta: só shapes, nenhuma memória real (o b0_8 tem ~0.8B parâmetros)
    with torch.device("meta"):
        return SoreLM(cfg)


def _linear_weights(model: SoreLM) -> int:
    roots = [model.blocks] + ([model.mtp] if model.mtp is not None else [])
    return sum(m.weight.numel() for root in roots for m in root.modules() if isinstance(m, nn.Linear))


def test_flops_at_least_6n():
    cfg = get_preset("b0_8")
    assert flops_per_token(cfg, 4096) >= 6 * _meta_model(cfg).num_params(exclude_embedding=True)


@pytest.mark.parametrize("preset", ["tiny", "s150m", "b0_8"])
def test_flops_monotonic_in_seq_len(preset):
    cfg = get_preset(preset)
    values = [flops_per_token(cfg, t) for t in (256, 1024, 2048, 4096, 8192)]
    assert all(b > a for a, b in zip(values, values[1:])), values


@pytest.mark.parametrize(
    "preset, overrides",
    [
        ("tiny", {}),
        ("tiny", {"mtp_layers": 0}),
        ("tiny", {"residual": "hyper"}),
        ("s150m", {}),
        ("b0_8", {}),
    ],
    ids=["tiny", "tiny_no_mtp", "tiny_hyper", "s150m", "b0_8"],
)
def test_matmul_params_matches_model(preset, overrides):
    # a conta analítica tem de bater com os pesos 2-D dos nn.Linear de blocks + mtp
    cfg = get_preset(preset, **overrides)
    assert matmul_params(cfg) == _linear_weights(_meta_model(cfg))


def test_flops_formula_terms():
    # b0_8: 18 GDN + 6 atenção no tronco, mais a atenção do MTP; cabeça usada 2x (LM + MTP)
    cfg = get_preset("b0_8")
    v_d = cfg.vocab_size * cfg.d_model
    attn = 6 * 7 * cfg.attn_heads * cfg.attn_head_dim
    gdn = 6 * 18 * cfg.gdn_v_heads * cfg.gdn_head_dim * cfg.gdn_head_dim * 2
    expected = 6 * (matmul_params(cfg) + 2 * v_d) + attn * 4096 + gdn
    assert flops_per_token(cfg, 4096) == expected
    # sem MTP: a cabeça entra uma vez e só as 6 atenções do tronco escalam com seq_len
    no_mtp = get_preset("b0_8", mtp_layers=0)
    slope = flops_per_token(no_mtp, 8192) - flops_per_token(no_mtp, 4096)
    assert slope == 6 * 6 * cfg.attn_heads * cfg.attn_head_dim * 4096
