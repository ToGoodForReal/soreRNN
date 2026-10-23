"""Referência em PyTorch puro do gated delta rule (convenção do FLA).

É o oráculo de corretude do kernel Triton/FLA e o backend de fallback em CPU.
Por ser o oráculo, segue a convenção do `naive_recurrent_gated_delta_rule` do FLA
ao pé da letra: decaimento antes do update, leitura da saída depois do update.
"""
import torch
from torch import Tensor


def _recurrence(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor, scale: float) -> Tensor:
    """Recorrência para um único trecho contínuo, com o estado começando em zero."""
    b, t, h, dk = q.shape
    dv = v.shape[-1]
    state = q.new_zeros(b, h, dk, dv)  # [B,H,Dk,Dv]
    outputs = []
    for i in range(t):
        state = state * g[:, i].exp()[..., None, None]
        delta = (v[:, i] - (state * k[:, i, ..., None]).sum(-2)) * beta[:, i, ..., None]
        state = state + k[:, i, ..., None] * delta[..., None, :]
        outputs.append(torch.einsum("bhd,bhdm->bhm", q[:, i] * scale, state))
    return torch.stack(outputs, dim=1)


def naive_gated_delta_rule(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    scale: float,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Gated delta rule sequencial.

    Shapes: `q,k [B,T,H,Dk]`, `v [B,T,H,Dv]`, `g,beta [B,T,H]`; retorna `[B,T,H,Dv]` no dtype de `v`.
    `g` é o log do decaimento (<= 0) e `beta` a taxa de escrita. Supõe `q` e `k` já normalizados em L2.
    A conta é feita em FP32 (FP64 é preservado, para permitir checagens numéricas exatas).

    Com `cu_seqlens` (`[S+1]`, de 0 a T), exige `B == 1` e zera o estado em cada início de documento.
    """
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(f"cu_seqlens exige B == 1, recebido B = {q.shape[0]}")

    out_dtype = v.dtype
    compute_dtype = torch.promote_types(out_dtype, torch.float32)
    q, k, v, g, beta = (x.to(compute_dtype) for x in (q, k, v, g, beta))

    if cu_seqlens is None:
        return _recurrence(q, k, v, g, beta, scale).to(out_dtype)

    bounds = cu_seqlens.tolist()
    t = q.shape[1]
    if bounds[0] != 0 or bounds[-1] != t:
        raise ValueError(f"cu_seqlens deve ir de 0 a T = {t}, recebido {bounds[0]}..{bounds[-1]}")
    # Poucos segmentos: um laço por documento mantém o autograd intacto e sem vazamento de estado.
    segments = [
        _recurrence(q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b], scale)
        for a, b in zip(bounds[:-1], bounds[1:])
    ]
    return torch.cat(segments, dim=1).to(out_dtype)
