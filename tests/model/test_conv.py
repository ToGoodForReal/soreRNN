"""Testes da conv causal depthwise curta com reset por documento."""
import pytest
import torch
import torch.nn.functional as F

from sore.model.conv import causal_conv1d_varlen
from sore.model.varlen import doc_ids_from_cu_seqlens


def _conv1d_oracle(x, weight):
    """Oráculo: conv1d depthwise causal com padding à esquerda (único uso de F.conv1d, só no teste)."""
    c, k = weight.shape
    t = x.shape[1]
    return F.conv1d(x.transpose(1, 2), weight[:, None, :], padding=k - 1, groups=c)[..., :t].transpose(1, 2)


@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("t", [1, 3, 10])
def test_conv_matches_conv1d_single_doc(k, t):
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(2, t, 5, generator=gen, dtype=torch.float64)
    weight = torch.randn(5, k, generator=gen, dtype=torch.float64)
    expected = _conv1d_oracle(x, weight)
    torch.testing.assert_close(causal_conv1d_varlen(x, weight, None), expected, atol=1e-12, rtol=0)
    # doc_ids de um único documento equivale a None
    one_doc = torch.zeros(t, dtype=torch.long)
    torch.testing.assert_close(causal_conv1d_varlen(x, weight, one_doc), expected, atol=1e-12, rtol=0)


def test_conv_tap_order():
    # weight[:, K-1] multiplica x_t e weight[:, K-1-j] multiplica x_{t-j}
    x = torch.zeros(1, 5, 1)
    x[0, 2, 0] = 1.0  # impulso em t=2
    weight = torch.tensor([[1.0, 10.0, 100.0]])  # K=3
    out = causal_conv1d_varlen(x, weight, None)
    assert out[0, :, 0].tolist() == [0.0, 0.0, 100.0, 10.0, 1.0]


def test_conv_no_leak():
    gen = torch.Generator().manual_seed(1)
    cu = torch.tensor([0, 4, 9], dtype=torch.long)
    doc_ids = doc_ids_from_cu_seqlens(cu, 9)
    x = torch.randn(1, 9, 3, generator=gen, dtype=torch.float64)
    weight = torch.randn(3, 4, generator=gen, dtype=torch.float64)
    base = causal_conv1d_varlen(x, weight, doc_ids)

    changed = x.clone()
    changed[:, :4] += torch.randn(1, 4, 3, generator=gen, dtype=torch.float64)  # só o doc 0
    out = causal_conv1d_varlen(changed, weight, doc_ids)

    torch.testing.assert_close(out[:, 4:], base[:, 4:], atol=0, rtol=0)
    assert not torch.allclose(out[:, :4], base[:, :4])


def test_conv_varlen_equals_separate_docs():
    gen = torch.Generator().manual_seed(2)
    cu = torch.tensor([0, 2, 3, 9], dtype=torch.long)  # inclui doc menor que K e doc de 1 token
    doc_ids = doc_ids_from_cu_seqlens(cu, 9)
    x = torch.randn(1, 9, 3, generator=gen, dtype=torch.float64)
    weight = torch.randn(3, 4, generator=gen, dtype=torch.float64)
    packed = causal_conv1d_varlen(x, weight, doc_ids)
    parts = [causal_conv1d_varlen(x[:, a:b], weight, None) for a, b in [(0, 2), (2, 3), (3, 9)]]
    torch.testing.assert_close(packed, torch.cat(parts, dim=1), atol=1e-12, rtol=0)


def test_conv_grads_match_conv1d_single_doc():
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(2, 7, 4, generator=gen, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(4, 3, generator=gen, dtype=torch.float64, requires_grad=True)
    cotangent = torch.randn(2, 7, 4, generator=gen, dtype=torch.float64)

    (causal_conv1d_varlen(x, weight, None) * cotangent).sum().backward()
    gx, gw = x.grad.clone(), weight.grad.clone()

    x.grad = None
    weight.grad = None
    (_conv1d_oracle(x, weight) * cotangent).sum().backward()

    torch.testing.assert_close(gx, x.grad, atol=1e-12, rtol=0)
    torch.testing.assert_close(gw, weight.grad, atol=1e-12, rtol=0)


def test_conv_grad_does_not_cross_documents():
    gen = torch.Generator().manual_seed(4)
    doc_ids = doc_ids_from_cu_seqlens(torch.tensor([0, 4, 9], dtype=torch.long), 9)
    x = torch.randn(1, 9, 3, generator=gen, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(3, 4, generator=gen, dtype=torch.float64, requires_grad=True)
    out = causal_conv1d_varlen(x, weight, doc_ids)
    out[:, 4:].sum().backward()  # só a saída do doc 1
    assert torch.count_nonzero(x.grad[:, :4]) == 0
    assert torch.count_nonzero(x.grad[:, 4:]) > 0


def test_conv_nonfinite_input_does_not_leak_across_documents():
    # Um NaN no doc 0 não pode contaminar o doc 1 (a máscara zera o termo, não multiplica por 0).
    doc_ids = doc_ids_from_cu_seqlens(torch.tensor([0, 4, 9], dtype=torch.long), 9)
    x = torch.randn(1, 9, 3, dtype=torch.float64)
    x[:, 3] = float("nan")
    weight = torch.randn(3, 4, dtype=torch.float64)
    out = causal_conv1d_varlen(x, weight, doc_ids)
    assert torch.isfinite(out[:, 4:]).all()
    assert torch.isnan(out[:, 3]).all()


def test_conv_preserves_dtype_and_shape():
    x = torch.randn(2, 6, 3, dtype=torch.bfloat16)
    weight = torch.randn(3, 4, dtype=torch.bfloat16)
    out = causal_conv1d_varlen(x, weight, None)
    assert out.shape == x.shape
    assert out.dtype == torch.bfloat16
