"""Testes da referência em PyTorch puro do gated delta rule."""
import pytest
import torch
import torch.nn.functional as F

from sore.model.gdn_ref import naive_gated_delta_rule

DK, DV, H = 8, 6, 2


def _inputs(b, t, dtype=torch.float64, seed=0, requires_grad=False):
    """q,k normalizados em L2 (premissa da função); g <= 0 (decaimento); beta em (0, 1)."""
    gen = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(b, t, H, DK, generator=gen, dtype=dtype), dim=-1)
    k = F.normalize(torch.randn(b, t, H, DK, generator=gen, dtype=dtype), dim=-1)
    v = torch.randn(b, t, H, DV, generator=gen, dtype=dtype)
    g = -torch.rand(b, t, H, generator=gen, dtype=dtype)
    beta = torch.rand(b, t, H, generator=gen, dtype=dtype)
    tensors = [q, k, v, g, beta]
    for x in tensors:
        x.requires_grad_(requires_grad)
    return tensors


def _cu(values):
    return torch.tensor(values, dtype=torch.long)


def test_single_step_closed_form():
    # T=1: o = scale * (q . k) * beta * v (o estado começa em zero, então o decaimento não importa)
    q, k, v, g, beta = _inputs(2, 1)
    scale = 0.37
    out = naive_gated_delta_rule(q, k, v, g, beta, scale=scale)
    qk = (q * k).sum(-1, keepdim=True)  # [B,1,H,1]
    expected = scale * qk * beta[..., None] * v
    assert out.shape == (2, 1, H, DV)
    torch.testing.assert_close(out, expected, atol=1e-12, rtol=0)


def test_two_steps_hand_computed():
    # Confere ordem das operações: decaimento antes do update, leitura depois do update.
    q, k, v, g, beta = _inputs(1, 2, seed=1)
    scale = 0.5
    out = naive_gated_delta_rule(q, k, v, g, beta, scale=scale)

    h = torch.zeros(1, H, DK, DV, dtype=torch.float64)
    expected = []
    for t in range(2):
        h = h * g[:, t].exp()[..., None, None]
        delta = (v[:, t] - torch.einsum("bhd,bhdm->bhm", k[:, t], h)) * beta[:, t, :, None]
        h = h + k[:, t, :, :, None] * delta[:, :, None, :]
        expected.append(torch.einsum("bhd,bhdm->bhm", q[:, t] * scale, h))
    torch.testing.assert_close(out, torch.stack(expected, dim=1), atol=1e-12, rtol=0)


def test_varlen_equals_separate():
    scale = DK**-0.5
    q, k, v, g, beta = _inputs(1, 12, dtype=torch.float32, seed=2)
    packed = naive_gated_delta_rule(q, k, v, g, beta, scale=scale, cu_seqlens=_cu([0, 5, 12]))
    first = naive_gated_delta_rule(q[:, :5], k[:, :5], v[:, :5], g[:, :5], beta[:, :5], scale=scale)
    second = naive_gated_delta_rule(q[:, 5:], k[:, 5:], v[:, 5:], g[:, 5:], beta[:, 5:], scale=scale)
    torch.testing.assert_close(packed, torch.cat([first, second], dim=1), atol=1e-6, rtol=0)


def test_varlen_resets_state_between_documents():
    # Sem o cu_seqlens o segundo documento enxerga o estado do primeiro; com ele não.
    scale = DK**-0.5
    q, k, v, g, beta = _inputs(1, 12, seed=3)
    cu = _cu([0, 5, 12])
    with_cu = naive_gated_delta_rule(q, k, v, g, beta, scale=scale, cu_seqlens=cu)
    without_cu = naive_gated_delta_rule(q, k, v, g, beta, scale=scale)
    torch.testing.assert_close(with_cu[:, :5], without_cu[:, :5], atol=1e-12, rtol=0)
    assert not torch.allclose(with_cu[:, 5:], without_cu[:, 5:], atol=1e-3)


def test_varlen_single_token_documents():
    scale = DK**-0.5
    q, k, v, g, beta = _inputs(1, 4, seed=4)
    packed = naive_gated_delta_rule(q, k, v, g, beta, scale=scale, cu_seqlens=_cu([0, 1, 2, 4]))
    parts = [
        naive_gated_delta_rule(q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b], scale=scale)
        for a, b in [(0, 1), (1, 2), (2, 4)]
    ]
    torch.testing.assert_close(packed, torch.cat(parts, dim=1), atol=1e-12, rtol=0)


def test_varlen_grad_wrt_q_has_no_state_leak():
    # O gradiente de q do empacotado deve ser o mesmo dos documentos separados.
    scale = DK**-0.5
    cu = _cu([0, 5, 12])
    weights = torch.randn(1, 12, H, DV, dtype=torch.float64, generator=torch.Generator().manual_seed(9))

    q, k, v, g, beta = _inputs(1, 12, seed=5, requires_grad=True)
    out = naive_gated_delta_rule(q, k, v, g, beta, scale=scale, cu_seqlens=cu)
    (out * weights).sum().backward()
    packed_grad = q.grad.clone()

    q2, k2, v2, g2, beta2 = _inputs(1, 12, seed=5, requires_grad=True)
    loss = 0.0
    for a, b in [(0, 5), (5, 12)]:
        part = naive_gated_delta_rule(
            q2[:, a:b], k2[:, a:b], v2[:, a:b], g2[:, a:b], beta2[:, a:b], scale=scale
        )
        loss = loss + (part * weights[:, a:b]).sum()
    loss.backward()

    torch.testing.assert_close(packed_grad, q2.grad, atol=1e-10, rtol=0)
    # os demais tensores também recebem gradiente (autograd intacto)
    for tensor in (k, v, g, beta):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_varlen_requires_batch_one():
    q, k, v, g, beta = _inputs(2, 6)
    with pytest.raises(ValueError, match="B == 1"):
        naive_gated_delta_rule(q, k, v, g, beta, scale=1.0, cu_seqlens=_cu([0, 3, 6]))


def test_varlen_rejects_cu_seqlens_not_covering_sequence():
    q, k, v, g, beta = _inputs(1, 6)
    with pytest.raises(ValueError, match="cu_seqlens"):
        naive_gated_delta_rule(q, k, v, g, beta, scale=1.0, cu_seqlens=_cu([0, 3, 5]))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_input_returns_input_dtype(dtype):
    q, k, v, g, beta = _inputs(1, 6, dtype=torch.float32, seed=6)
    out = naive_gated_delta_rule(
        q.to(dtype), k.to(dtype), v.to(dtype), g.to(dtype), beta.to(dtype), scale=DK**-0.5
    )
    assert out.dtype == dtype
    # a conta é feita em FP32: perto do resultado FP32 até o erro de arredondamento da entrada
    ref = naive_gated_delta_rule(
        q.to(dtype).float(), k.to(dtype).float(), v.to(dtype).float(),
        g.to(dtype).float(), beta.to(dtype).float(), scale=DK**-0.5,
    )
    torch.testing.assert_close(out.float(), ref.to(dtype).float(), atol=0, rtol=0)
