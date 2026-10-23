"""Testes dos helpers de sequências empacotadas (cu_seqlens)."""
import pytest
import torch

from sore.model.varlen import doc_ids_from_cu_seqlens, positions_from_cu_seqlens


def _cu(values, **kw):
    return torch.tensor(values, dtype=torch.long, **kw)


def test_doc_ids():
    out = doc_ids_from_cu_seqlens(_cu([0, 3, 5]), 5)
    assert out.tolist() == [0, 0, 0, 1, 1]
    assert out.dtype == torch.long


def test_positions():
    out = positions_from_cu_seqlens(_cu([0, 3, 5]), 5)
    assert out.tolist() == [0, 1, 2, 0, 1]
    assert out.dtype == torch.long


def test_single_token_documents():
    cu = _cu([0, 1, 2, 5])
    assert doc_ids_from_cu_seqlens(cu, 5).tolist() == [0, 1, 2, 2, 2]
    assert positions_from_cu_seqlens(cu, 5).tolist() == [0, 0, 0, 1, 2]


def test_int32_cu_seqlens_returns_long():
    cu = torch.tensor([0, 3, 5], dtype=torch.int32)
    ids = doc_ids_from_cu_seqlens(cu, 5)
    pos = positions_from_cu_seqlens(cu, 5)
    assert ids.dtype == pos.dtype == torch.long
    assert ids.tolist() == [0, 0, 0, 1, 1]
    assert pos.tolist() == [0, 1, 2, 0, 1]


@pytest.mark.cuda
def test_outputs_live_on_cu_seqlens_device():
    cu = _cu([0, 3, 5], device="cuda")
    ids = doc_ids_from_cu_seqlens(cu, 5)
    pos = positions_from_cu_seqlens(cu, 5)
    assert ids.device == pos.device == cu.device
    assert ids.cpu().tolist() == [0, 0, 0, 1, 1]
    assert pos.cpu().tolist() == [0, 1, 2, 0, 1]
