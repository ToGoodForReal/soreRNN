"""Testes de RMSNorm e SwiGLU."""
import torch

from sore.model.layers import RMSNorm, SwiGLU


def test_rmsnorm_matches_reference_and_inits_to_ones():
    norm = RMSNorm(16, eps=1e-6)
    assert torch.equal(norm.weight, torch.ones(16))
    x = torch.randn(2, 5, 16)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(norm(x), ref)


def test_rmsnorm_preserves_input_dtype():
    norm = RMSNorm(16, eps=1e-6)
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    y = norm(x)
    assert y.dtype == torch.bfloat16 and y.shape == x.shape


def test_swiglu_shape_formula_and_no_bias():
    ffn = SwiGLU(8, 24)
    assert ffn.gate_proj.bias is None and ffn.up_proj.bias is None and ffn.down_proj.bias is None
    assert tuple(ffn.gate_proj.weight.shape) == (24, 8)
    assert tuple(ffn.up_proj.weight.shape) == (24, 8)
    assert tuple(ffn.down_proj.weight.shape) == (8, 24)
    x = torch.randn(2, 4, 8)
    ref = ffn.down_proj(torch.nn.functional.silu(ffn.gate_proj(x)) * ffn.up_proj(x))
    out = ffn(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out, ref)
