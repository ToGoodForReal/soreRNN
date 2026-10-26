"""Loss de LM em chunks: os logits `[N, V]` nunca ficam guardados para o backward."""
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

IGNORE_INDEX = -100


def _ce_sum(hidden: Tensor, weight: Tensor, labels: Tensor) -> Tensor:
    logits = (hidden @ weight.T).float()
    return F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX, reduction="sum")


class ChunkCrossEntropy(nn.Module):
    """Soma da cross-entropy fp32 de um chunk: `hidden [n,D]`, `weight [V,D]`, `labels [n]`.

    É um módulo (sem parâmetros) só para o `compile_model` poder compilá-lo in-place. Compilado, o
    Inductor funde matmul, log-softmax e o backward sem materializar logits fp32 e log-probs
    `n x V` separados.
    """

    def forward(self, hidden: Tensor, weight: Tensor, labels: Tensor) -> Tensor:
        return _ce_sum(hidden, weight, labels)


def chunked_lm_loss(
    hidden: Tensor,
    weight: Tensor,
    labels: Tensor,
    chunk_size: int = 8192,
    ce: Callable[[Tensor, Tensor, Tensor], Tensor] = _ce_sum,
) -> tuple[Tensor, Tensor]:
    """Soma da cross-entropy (fp32) e número de tokens válidos (`labels != -100`).

    A contagem é um tensor int64 0-dim no device de `labels`, não um `int`: `int(tensor)` leria o
    valor na CPU e faria a GPU esperar o host (duas vezes por micro-batch, com o MTP).

    `hidden [N,D]`, `weight [V,D]` (peso da cabeça de LM) e `labels [N]`. Cada chunk de até
    `chunk_size` linhas roda `ce` (a soma da CE do chunk, ex. `ChunkCrossEntropy`) sob
    `checkpoint`: o forward descarta os logits do chunk e o backward os recalcula, então a memória
    guardada fica em `chunk_size x V` por vez, nunca `N x V`. Sem tokens válidos a soma é zero,
    ainda ligada ao grafo (gradiente nulo).
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size deve ser positivo; recebi {chunk_size}")
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, hidden.shape[0], chunk_size):
        end = start + chunk_size
        total = total + checkpoint(ce, hidden[start:end], weight, labels[start:end], use_reentrant=False)
    return total, (labels != IGNORE_INDEX).sum()


def fused_lm_loss(hidden: Tensor, weight: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Soma da cross-entropy (fp32) e número de tokens válidos (`labels != -100`) via kernel fundido do FLA.

    Fundindo o linear final com a cross-entropy em chunks internos no kernel Triton, os logits
    `N x V` nunca são materializados em memória global (nem no forward, nem no backward). Segue o
    mesmo contrato de `chunked_lm_loss`: soma fp32 0-dim e contagem int64 0-dim no device de `labels`.
    Sob autocast, `hidden` e `weight` são convertidos para o mesmo dtype de autocast, com acumulação
    de gradientes do peso em fp32. Sem tokens válidos a soma é zero conectada ao grafo (gradiente nulo).
    """
    from fla.modules.fused_linear_cross_entropy import fused_linear_cross_entropy_loss

    device_type = "cuda" if hidden.is_cuda else "cpu"
    if torch.is_autocast_enabled(device_type):
        autocast_dtype = torch.get_autocast_dtype(device_type)
        if hidden.dtype != autocast_dtype:
            hidden = hidden.to(autocast_dtype)
        if weight.dtype != autocast_dtype:
            weight = weight.to(autocast_dtype)
    elif hidden.dtype != weight.dtype:
        weight = weight.to(hidden.dtype)

    total = fused_linear_cross_entropy_loss(
        hidden,
        labels,
        weight,
        ignore_index=IGNORE_INDEX,
        reduction="sum",
        accumulate_grad_in_fp32=True,
    )
    return total, (labels != IGNORE_INDEX).sum()

