"""GatedAttention: o mixer de atenção (1 de cada 4 camadas do soreQ4).

GQA com RoPE parcial, q/k-norm e gate sigmoidal na saída. Dois backends com os mesmos pesos:
- `sdpa`: `F.scaled_dot_product_attention` com máscara booleana densa (CPU, testes e oráculo);
- `flex`: `flex_attention` com `BlockMask` de documento-causal (treino em GPU, sem matriz T x T densa).

O `flex_attention` só é importado dentro do backend `flex`, para que este módulo carregue sem ele.
"""
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sore.model.config import ModelConfig
from sore.model.layers import RMSNorm
from sore.model.rope import PartialRotary
from sore.model.varlen import doc_ids_from_cu_seqlens, positions_from_cu_seqlens

_compiled_flex = None  # torch.compile(flex_attention), criado na primeira chamada em CUDA


def _get_flex_attention(on_cuda: bool):
    """`flex_attention` compilado em CUDA (sem compilar ele materializa a matriz de scores inteira)."""
    from torch.nn.attention.flex_attention import flex_attention

    if not on_cuda:
        return flex_attention
    global _compiled_flex
    if _compiled_flex is None:
        _compiled_flex = torch.compile(flex_attention)
    return _compiled_flex


class GatedAttention(nn.Module):
    """`o_proj(attn * sigmoid(gate_proj(x)))`, com atenção causal GQA e RoPE parcial.

    `forward(x [B,T,D], cu_seqlens)`: com `cu_seqlens` (`[S+1]`, exige `B == 1`) a atenção é causal
    dentro de cada documento e as posições do RoPE reiniciam em 0 em cada um; sem ele, cada linha
    do batch é um documento.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.attn_backend not in ("sdpa", "flex"):
            raise ValueError(f"attn_backend desconhecido: {cfg.attn_backend!r}")
        self.backend = cfg.attn_backend
        self.heads = cfg.attn_heads
        self.kv_heads = cfg.attn_kv_heads
        self.head_dim = cfg.attn_head_dim
        d = cfg.d_model

        self.q_proj = nn.Linear(d, self.heads * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(d, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.norm_eps)
        self.rotary = PartialRotary(self.head_dim, cfg.rope_partial, cfg.rope_theta)
        self.o_proj = nn.Linear(self.heads * self.head_dim, d, bias=False)

    def forward(self, x: Tensor, cu_seqlens: Tensor | None = None) -> Tensor:
        b, t, _ = x.shape
        if cu_seqlens is not None and b != 1:
            raise ValueError(f"cu_seqlens exige batch 1 (layout empacotado [1, N]); recebi B={b}")

        q = self.q_norm(self.q_proj(x).view(b, t, self.heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(b, t, self.kv_heads, self.head_dim))
        v = self.v_proj(x).view(b, t, self.kv_heads, self.head_dim)

        if cu_seqlens is None:
            doc_ids = None
            positions = torch.arange(t, device=x.device)
        else:
            doc_ids = doc_ids_from_cu_seqlens(cu_seqlens, t)
            positions = positions_from_cu_seqlens(cu_seqlens, t)
        q, k = self.rotary(q, k, positions)

        # [B,T,H,Dh] -> [B,H,T,Dh]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.backend == "sdpa":
            o = self._sdpa(q, k, v, doc_ids)
        else:
            o = self._flex(q, k, v, doc_ids)

        o = o.transpose(1, 2).reshape(b, t, self.heads * self.head_dim)
        return self.o_proj(o * torch.sigmoid(self.gate_proj(x)))

    @staticmethod
    def _sdpa(q: Tensor, k: Tensor, v: Tensor, doc_ids: Tensor | None) -> Tensor:
        if doc_ids is None:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        t = q.shape[2]
        # True = pode atender: causal e mesmo documento
        causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
        mask = causal & (doc_ids[:, None] == doc_ids[None, :])
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)

    @staticmethod
    def _flex(q: Tensor, k: Tensor, v: Tensor, doc_ids: Tensor | None) -> Tensor:
        from torch.nn.attention.flex_attention import create_block_mask

        t = q.shape[2]
        if doc_ids is None:

            def mask_mod(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx

        else:

            def mask_mod(b, h, q_idx, kv_idx):
                return (q_idx >= kv_idx) & (doc_ids[q_idx] == doc_ids[kv_idx])

        block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=t, KV_LEN=t, device=q.device)
        flex_attention = _get_flex_attention(q.is_cuda)
        return flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)
