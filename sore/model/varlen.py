"""Helpers para o layout empacotado: `cu_seqlens` -> id do documento e posição dentro dele.

Pré-condição (não verificada, para não forçar sincronização na GPU): `cu_seqlens`
tem shape `[S+1]`, começa em 0, termina em `n` e é estritamente crescente.
"""
import torch
from torch import Tensor


def doc_ids_from_cu_seqlens(cu_seqlens: Tensor, n: int) -> Tensor:
    """Id (0..S-1) do documento de cada posição. Retorna `LongTensor[n]` no device de `cu_seqlens`."""
    positions = torch.arange(n, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
    # right=True: a posição que coincide com uma fronteira já pertence ao documento seguinte
    return torch.searchsorted(cu_seqlens[1:].contiguous(), positions, right=True).long()


def positions_from_cu_seqlens(cu_seqlens: Tensor, n: int) -> Tensor:
    """Posição de cada token dentro do seu documento. Retorna `LongTensor[n]` no device de `cu_seqlens`."""
    doc_ids = doc_ids_from_cu_seqlens(cu_seqlens, n)
    starts = cu_seqlens.long()[doc_ids]
    return torch.arange(n, device=cu_seqlens.device, dtype=torch.long) - starts
