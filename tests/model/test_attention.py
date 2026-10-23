"""Testes da GatedAttention (GQA, RoPE parcial, máscara de documento) e do PartialRotary."""
import pytest
import torch

from sore.model.attention import GatedAttention
from sore.model.config import get_preset
from sore.model.rope import PartialRotary


def _layer(device="cpu", seed=0, **overrides):
    torch.manual_seed(seed)
    return GatedAttention(get_preset("tiny", **overrides)).to(device)


def _cu(values, device="cpu"):
    return torch.tensor(values, dtype=torch.long, device=device)


def test_rope_identity_at_zero():
    rope = PartialRotary(head_dim=64, partial=0.25, theta=1e7)
    q = torch.randn(2, 5, 2, 64)
    k = torch.randn(2, 5, 1, 64)
    q_out, k_out = rope(q, k, torch.zeros(5, dtype=torch.long))
    torch.testing.assert_close(q_out, q, atol=0, rtol=0)
    torch.testing.assert_close(k_out, k, atol=0, rtol=0)


def test_rope_untouched_tail():
    rope = PartialRotary(head_dim=64, partial=0.25, theta=1e7)
    rot_dim = rope.rot_dim
    assert rot_dim == 16
    q = torch.randn(2, 7, 2, 64)
    k = torch.randn(2, 7, 1, 64)
    q_out, k_out = rope(q, k, torch.arange(7))
    assert torch.equal(q_out[..., rot_dim:], q[..., rot_dim:])
    assert torch.equal(k_out[..., rot_dim:], k[..., rot_dim:])
    # a parte girada muda (de verdade) fora da posição 0, mas preserva a norma
    assert not torch.allclose(q_out[:, 1:, :, :rot_dim], q[:, 1:, :, :rot_dim])
    torch.testing.assert_close(
        q_out[..., :rot_dim].norm(dim=-1), q[..., :rot_dim].norm(dim=-1), atol=1e-5, rtol=1e-5
    )


def test_rope_is_relative():
    # <rot(q, m), rot(k, n)> só depende de m - n
    rope = PartialRotary(head_dim=64, partial=0.25, theta=1e7)
    q = torch.randn(1, 1, 1, 64)
    k = torch.randn(1, 1, 1, 64)

    def score(m, n):
        qr, _ = rope(q, k, torch.tensor([m]))
        _, kr = rope(q, k, torch.tensor([n]))
        return (qr * kr).sum()

    torch.testing.assert_close(score(10, 4), score(106, 100), atol=1e-4, rtol=1e-4)


def test_rope_rejects_odd_rot_dim():
    with pytest.raises(ValueError):
        PartialRotary(head_dim=60, partial=0.25, theta=1e7)  # int(60 * .25) = 15


def test_shapes_finite():
    cfg = get_preset("tiny")
    layer = _layer()
    h, hkv, dh = cfg.attn_heads, cfg.attn_kv_heads, cfg.attn_head_dim
    assert layer.q_proj.weight.shape == layer.gate_proj.weight.shape == (h * dh, cfg.d_model)
    assert layer.k_proj.weight.shape == layer.v_proj.weight.shape == (hkv * dh, cfg.d_model)
    assert layer.q_norm.weight.shape == layer.k_norm.weight.shape == (dh,)
    assert layer.o_proj.weight.shape == (cfg.d_model, h * dh)
    assert all(m.bias is None for m in layer.modules() if isinstance(m, torch.nn.Linear))

    x = torch.randn(2, 10, cfg.d_model, requires_grad=True)
    out = layer(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.isfinite(out).all()

    out.square().mean().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_causal():
    layer = _layer()
    x = torch.randn(1, 16, 128)
    x2 = x.clone()
    x2[:, 9] += torch.randn(128)
    out, out2 = layer(x), layer(x2)
    torch.testing.assert_close(out[:, :9], out2[:, :9], atol=1e-6, rtol=0)
    assert not torch.allclose(out[:, 9:], out2[:, 9:], atol=1e-4)


def test_packed_equals_separate():
    layer = _layer()
    x = torch.randn(1, 20, 128)
    packed = layer(x, cu_seqlens=_cu([0, 7, 20]))
    separate = torch.cat([layer(x[:, :7]), layer(x[:, 7:])], dim=1)
    torch.testing.assert_close(packed, separate, atol=1e-5, rtol=0)
    # sem cu_seqlens o segundo documento enxerga o primeiro (e as posições não reiniciam)
    assert not torch.allclose(layer(x)[:, 7:], separate[:, 7:], atol=1e-3)


def test_no_cross_document_attention():
    # perturbar um documento anterior não muda a saída do seguinte
    layer = _layer()
    x = torch.randn(1, 20, 128)
    x2 = x.clone()
    x2[:, :7] += torch.randn(1, 7, 128)
    cu = _cu([0, 7, 20])
    torch.testing.assert_close(layer(x, cu)[:, 7:], layer(x2, cu)[:, 7:], atol=1e-6, rtol=0)


def test_batch_rows_are_independent_documents():
    # Sem cu_seqlens e com B > 1, cada linha do batch é um documento.
    layer = _layer()
    x = torch.randn(3, 9, 128)
    batched = layer(x)
    rows = torch.cat([layer(x[i : i + 1]) for i in range(3)], dim=0)
    torch.testing.assert_close(batched, rows, atol=1e-5, rtol=0)


def test_cu_seqlens_requires_batch_one():
    layer = _layer()
    with pytest.raises(ValueError):
        layer(torch.randn(2, 8, 128), cu_seqlens=_cu([0, 8]))


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        GatedAttention(get_preset("tiny", attn_backend="nope"))


@pytest.mark.filterwarnings("ignore:flex_attention called without torch.compile:UserWarning")
def test_flex_matches_sdpa_cpu():
    # o flex sem compilação roda em CPU (só forward: não há backward de flex em CPU)
    sdpa = _layer(attn_backend="sdpa")
    flex = _layer(attn_backend="flex")
    flex.load_state_dict(sdpa.state_dict())
    x = torch.randn(1, 40, 128)
    cu = _cu([0, 15, 40])
    with torch.no_grad():
        torch.testing.assert_close(flex(x, cu), sdpa(x, cu), atol=1e-4, rtol=0)
        torch.testing.assert_close(flex(x), sdpa(x), atol=1e-4, rtol=0)


@pytest.mark.cuda
def test_flex_matches_sdpa():
    sdpa = _layer(device="cuda", attn_backend="sdpa")
    flex = _layer(device="cuda", attn_backend="flex")
    flex.load_state_dict(sdpa.state_dict())
    torch.manual_seed(1)
    x = torch.randn(1, 300, 128, device="cuda")
    cu = _cu([0, 120, 300], device="cuda")
    with torch.no_grad():
        ref = sdpa(x, cu)
        out = flex(x, cu)
    max_diff = (out - ref).abs().max().item()
    print(f"flex vs sdpa: max abs diff = {max_diff:.3e}")
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=0)


@pytest.mark.cuda
def test_flex_backward_matches_sdpa():
    # o caminho de treino: gradientes dos pesos e da entrada iguais nos dois backends
    sdpa = _layer(device="cuda", attn_backend="sdpa")
    flex = _layer(device="cuda", attn_backend="flex")
    flex.load_state_dict(sdpa.state_dict())
    torch.manual_seed(2)
    x = torch.randn(1, 300, 128, device="cuda")
    cu = _cu([0, 120, 300], device="cuda")
    grads = []
    for layer in (sdpa, flex):
        xi = x.clone().requires_grad_()
        layer(xi, cu).square().mean().backward()
        grads.append([xi.grad, *(p.grad for p in layer.parameters())])
    for g_ref, g_flex in zip(*grads):
        torch.testing.assert_close(g_flex, g_ref, atol=2e-3, rtol=1e-3)
