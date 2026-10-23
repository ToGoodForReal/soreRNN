"""Multi-token prediction de profundidade 1 (estilo DeepSeek-V3): na posição t, prevê o token t+2.

O módulo combina o hidden final do tronco `h[t]` com o embedding do token t+1 (`emb_next[t]`),
passa por um Block de atenção (residual plain) e por uma norma final. A cabeça de LM é a do
tronco (peso compartilhado). Nada atravessa documentos: `emb_next` e `mtp_labels` param na última
posição de cada documento e o Block recebe `cu_seqlens`.
"""
import dataclasses

import torch
from torch import Tensor, nn

from sore.model.block import Block
from sore.model.config import ModelConfig
from sore.model.layers import RMSNorm
from sore.model.loss import IGNORE_INDEX


def _shift_left_in_doc(x: Tensor, cu_seqlens: Tensor | None, fill: float | int) -> Tensor:
    """`out[:, t] = x[:, t+1]` se `t+1` está no mesmo documento (na mesma linha, sem `cu_seqlens`), senão `fill`."""
    pad = x.new_full((x.shape[0], 1, *x.shape[2:]), fill)
    out = torch.cat([x[:, 1:], pad], dim=1)
    if cu_seqlens is not None:
        # última posição de cada documento (layout empacotado [1, N])
        out = out.index_fill(1, cu_seqlens[1:].long() - 1, fill)
    return out


def mtp_labels(labels: Tensor, cu_seqlens: Tensor | None = None) -> Tensor:
    """Alvos do MTP: `out[t] = labels[t+1]` se `t+1` está no mesmo documento, senão `-100`."""
    return _shift_left_in_doc(labels, cu_seqlens, IGNORE_INDEX)


def next_token_embeddings(emb: Tensor, cu_seqlens: Tensor | None = None) -> Tensor:
    """`emb_next[t] = emb[t+1]` dentro do documento; zero na última posição de cada documento."""
    return _shift_left_in_doc(emb, cu_seqlens, 0.0)


class MTPModule(nn.Module):
    """`final_norm(block(proj(cat(norm_h(h), norm_e(emb_next)))))`, com `h` e `emb_next` em `[B,T,D]`."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.norm_h = RMSNorm(d, eps=cfg.norm_eps)
        self.norm_e = RMSNorm(d, eps=cfg.norm_eps)
        self.proj = nn.Linear(2 * d, d, bias=False)
        self.block = Block(dataclasses.replace(cfg, residual="plain"), layer_idx=cfg.n_layers, mixer_type="attn")
        self.final_norm = RMSNorm(d, eps=cfg.norm_eps)

    def forward(self, h: Tensor, emb_next: Tensor, cu_seqlens: Tensor | None = None, block_mask=None) -> Tensor:
        # proj roda em bf16 sob autocast; .float() mantém o stream residual do MTP em fp32, como o do tronco
        x = self.proj(torch.cat([self.norm_h(h), self.norm_e(emb_next)], dim=-1)).float()
        return self.final_norm(self.block(x, cu_seqlens, block_mask))
