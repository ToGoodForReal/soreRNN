"""Conv causal depthwise curta, com reset por documento (q/k/v do GatedDeltaNet)."""
import torch.nn.functional as F
from torch import Tensor


def causal_conv1d_varlen(x: Tensor, weight: Tensor, doc_ids: Tensor | None) -> Tensor:
    """`y_t = sum_j weight[:, K-1-j] * x_{t-j}`, ignorando termos de outro documento ou antes do início.

    `x [B,T,C]`, `weight [C,K]`, `doc_ids LongTensor[T]` (`None` = um documento por linha do batch).
    Implementada como soma de K deslocamentos mascarados, sem `F.conv1d`, para ser a mesma em CPU e GPU.
    """
    t = x.shape[1]
    k = weight.shape[1]
    out = x * weight[:, k - 1]
    for j in range(1, min(k, t)):
        # termo para t >= j: x_{t-j} * weight[:, K-1-j]
        term = x[:, : t - j] * weight[:, k - 1 - j]
        if doc_ids is not None:
            same_doc = doc_ids[j:] == doc_ids[: t - j]
            # masked_fill (e não multiplicar por 0): um NaN/inf em outro documento não pode vazar
            term = term.masked_fill(~same_doc[None, :, None], 0.0)
        out = out + F.pad(term, (0, 0, j, 0))
    return out
