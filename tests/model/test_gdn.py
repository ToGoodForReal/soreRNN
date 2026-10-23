"""Testes da camada GatedDeltaNet (backends `torch` de referência e `fla`)."""
import pytest
import torch

from sore.model.config import get_preset
from sore.model.gdn import GatedDeltaNet


def _layer(backend="torch", device="cpu", seed=0):
    torch.manual_seed(seed)
    return GatedDeltaNet(get_preset("tiny", gdn_backend=backend)).to(device)


def _cu(values, device="cpu"):
    return torch.tensor(values, dtype=torch.long, device=device)


def test_shapes_finite():
    cfg = get_preset("tiny")
    layer = _layer()
    hqk, hv, dh, k = cfg.gdn_qk_heads, cfg.gdn_v_heads, cfg.gdn_head_dim, cfg.conv_kernel
    assert layer.q_proj.weight.shape == layer.k_proj.weight.shape == (hqk * dh, cfg.d_model)
    assert layer.v_proj.weight.shape == layer.g_proj.weight.shape == (hv * dh, cfg.d_model)
    assert layer.a_proj.weight.shape == layer.b_proj.weight.shape == (hv, cfg.d_model)
    assert layer.conv_q.shape == layer.conv_k.shape == (hqk * dh, k)
    assert layer.conv_v.shape == (hv * dh, k)
    assert layer.A_log.shape == layer.dt_bias.shape == (hv,)
    assert layer.o_proj.weight.shape == (cfg.d_model, hv * dh)
    assert all(m.bias is None for m in layer.modules() if isinstance(m, torch.nn.Linear))

    x = torch.randn(2, 10, cfg.d_model, requires_grad=True)
    out = layer(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.isfinite(out).all()

    out.square().mean().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_packed_equals_separate():
    layer = _layer()
    x = torch.randn(1, 20, 128)
    packed = layer(x, cu_seqlens=_cu([0, 7, 20]))
    separate = torch.cat([layer(x[:, :7]), layer(x[:, 7:])], dim=1)
    torch.testing.assert_close(packed, separate, atol=1e-5, rtol=0)
    # sem cu_seqlens o segundo documento enxerga o estado e a conv do primeiro
    assert not torch.allclose(layer(x)[:, 7:], separate[:, 7:], atol=1e-3)


def test_batch_rows_are_independent_documents():
    # Sem cu_seqlens e com B > 1, cada linha do batch é um documento.
    layer = _layer()
    x = torch.randn(3, 9, 128)
    batched = layer(x)
    rows = torch.cat([layer(x[i : i + 1]) for i in range(3)], dim=0)
    torch.testing.assert_close(batched, rows, atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    "backend",
    ["torch", pytest.param("fla", marks=pytest.mark.fla)],
)
def test_extreme_inputs_finite(backend):
    device = "cuda" if backend == "fla" else "cpu"
    layer = _layer(backend, device)
    x = (torch.randn(1, 64, 128, device=device) * 100).requires_grad_(True)
    out = layer(x, cu_seqlens=_cu([0, 30, 64], device))
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.fla
def test_fla_matches_torch():
    # Mesmos pesos nos dois backends; fp32 com documentos de 1 token e um maior que o chunk (64) do FLA.
    fla_layer = _layer("fla", "cuda")
    ref_layer = _layer("torch", "cuda", seed=1)
    ref_layer.load_state_dict(fla_layer.state_dict())

    gen = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(1, 257, 128, device="cuda", generator=gen)
    cotangent = torch.randn(1, 257, 128, device="cuda", generator=gen)
    cu = _cu([0, 1, 2, 100, 257], "cuda")

    results = {}
    for name, layer in (("fla", fla_layer), ("torch", ref_layer)):
        xi = x.clone().requires_grad_(True)
        out = layer(xi, cu_seqlens=cu)
        (out * cotangent).sum().backward()
        results[name] = (out.detach(), xi.grad, layer.q_proj.weight.grad)

    (out_fla, gx_fla, gq_fla), (out_ref, gx_ref, gq_ref) = results["fla"], results["torch"]
    torch.testing.assert_close(out_fla, out_ref, atol=1e-3, rtol=0)
    # Em fp32 os tl.dot do FLA usam TF32: erro máximo medido ~1.2e-3 * max|ref| nos gradientes
    # (com TRITON_F32_DEFAULT=ieee cai para ~3e-5 * max|ref|, ou seja, não é erro de convenção).
    # atol = 5e-3 * max|ref| cobre os elementos perto de zero, onde rtol não tem sentido;
    # um erro de convenção (scale, g, beta, reset) dá erro relativo O(1).
    for actual, expected in ((gx_fla, gx_ref), (gq_fla, gq_ref)):
        atol = 5e-3 * expected.abs().max().item()
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=atol)
