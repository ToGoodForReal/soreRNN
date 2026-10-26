"""Políticas e funções auxiliares de activation checkpointing do soreQ4.

Oferece suporte a:
- `"full"`: o checkpoint padrão do PyTorch envolvendo o Block inteiro;
- `"ffn"`: checkpoint apenas de `norm2 + FFN` (SwiGLU) dentro do Block;
- `"selective"`: checkpoint seletivo via `torch.utils.checkpoint.create_selective_checkpoint_contexts`,
  salvando saídas de matmul (`aten.mm`, `aten.addmm`, `aten.bmm`, `aten._scaled_mm`) e da atenção,
  e recomputando ativações baratas (normas, SiLU, gates).

Nota técnica sobre kernels do FLA:
Os operadores do flash-linear-attention (como `chunk_gated_delta_rule`) são implementados como
`torch.autograd.Function` diretos sobre kernels Triton e não passam pelo dispatcher de operadores
ATen (`torch.ops.aten`). Portanto, sob a política de checkpoint seletivo do dispatcher ATen,
o kernel do GDN não intercepta MUST_SAVE e é sempre recomputado durante o backward.
Por outro lado, todas as projeções lineares (`q_proj`, `k_proj`, `v_proj`, `g_proj`, `a_proj`,
`b_proj`, `o_proj`, bem como as projeções lineares da SwiGLU `w1`, `w2`, `w3`) são matmuls
interceptados pelo dispatcher e têm suas ativações salvas na memória.
"""
from __future__ import annotations

import functools
import torch
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts


def checkpointed_blocks(n_layers: int, frac: float) -> frozenset[int]:
    """Conjunto de índices de Blocks do tronco sob checkpoint.

    Seleciona `round(frac * n_layers)` blocos espalhados por igual ao longo de `n_layers`.
    `frac` deve estar no intervalo (0, 1].
    - `frac=1.0` seleciona todos os blocos `0 .. n_layers-1`;
    - `(24, 0.5)` seleciona 12 índices sem vizinhos consecutivos.
    """
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"checkpoint_frac deve estar em (0, 1]; recebi {frac}")
    if n_layers < 1:
        raise ValueError(f"n_layers deve ser >= 1; recebi {n_layers}")
    k = round(frac * n_layers)
    if k == 0:
        return frozenset()
    if k == n_layers:
        return frozenset(range(n_layers))
    step = n_layers / k
    return frozenset(int(i * step) for i in range(k))


# Operadores de matmul do ATen salvos pela política seletiva
_MATMUL_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
}
if hasattr(torch.ops.aten, "_scaled_mm"):
    _MATMUL_OPS.add(torch.ops.aten._scaled_mm.default)

# Operadores de atenção do ATen
_ATTN_OPS = set()
for _name in (
    "scaled_dot_product_attention",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_cudnn_attention",
):
    if hasattr(torch.ops.aten, _name):
        _ATTN_OPS.add(getattr(torch.ops.aten, _name).default)

_OPS_TO_SAVE = _MATMUL_OPS | _ATTN_OPS


def _is_attention_op(op) -> bool:
    name = str(op)
    return "attention" in name or "flex" in name


def selective_policy_fn(ctx, op, *args, **kwargs) -> CheckpointPolicy:
    """Política seletiva: salva saídas de matmul e atenção, recomputa o resto."""
    if op in _OPS_TO_SAVE or _is_attention_op(op):
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def selective_checkpoint_context_fn():
    """Função de contexto de checkpoint seletivo para passar a `torch.utils.checkpoint.checkpoint`."""
    return create_selective_checkpoint_contexts(selective_policy_fn)
