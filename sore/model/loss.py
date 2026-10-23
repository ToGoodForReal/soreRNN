"""Loss de LM em chunks: os logits `[N, V]` nunca ficam guardados para o backward."""
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

IGNORE_INDEX = -100


def _ce_sum(hidden: Tensor, weight: Tensor, labels: Tensor) -> Tensor:
    logits = (hidden @ weight.T).float()
    return F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX, reduction="sum")


def chunked_lm_loss(
    hidden: Tensor, weight: Tensor, labels: Tensor, chunk_size: int = 8192
) -> tuple[Tensor, int]:
    """Soma da cross-entropy (fp32) e número de tokens válidos (`labels != -100`).

    `hidden [N,D]`, `weight [V,D]` (peso da cabeça de LM) e `labels [N]`. Cada chunk de até
    `chunk_size` linhas roda sob `checkpoint`: o forward descarta os logits do chunk e o backward
    os recalcula, então a memória guardada fica em `chunk_size x V` por vez, nunca `N x V`.
    Sem tokens válidos a soma é zero, ainda ligada ao grafo (gradiente nulo).
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size deve ser positivo; recebi {chunk_size}")
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, hidden.shape[0], chunk_size):
        end = start + chunk_size
        total = total + checkpoint(
            _ce_sum, hidden[start:end], weight, labels[start:end], use_reentrant=False
        )
    return total, int((labels != IGNORE_INDEX).sum())
