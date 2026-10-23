"""GatedDeltaNet: o mixer recorrente (3 de cada 4 camadas do soreQ4).

Dois backends com os mesmos pesos e o mesmo resultado:
- `fla`: kernel Triton `chunk_gated_delta_rule` do flash-linear-attention (treino em GPU);
- `torch`: referência sequencial `naive_gated_delta_rule` (fallback em CPU e oráculo).

O FLA só é importado dentro do backend `fla`, para que este módulo carregue sem Triton.
"""
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sore.model.config import ModelConfig
from sore.model.conv import causal_conv1d_varlen
from sore.model.gdn_ref import naive_gated_delta_rule
from sore.model.layers import RMSNorm
from sore.model.varlen import doc_ids_from_cu_seqlens

_DT_MIN, _DT_MAX = 1e-3, 1e-1


class GatedDeltaNet(nn.Module):
    """`o_proj(RMSNorm(o) * silu(g_proj(x)))`, com `o` vindo do gated delta rule sobre q/k/v pós-conv.

    `forward(x [B,T,D], cu_seqlens)`: com `cu_seqlens` (`[S+1]`, exige `B == 1`) a conv e o estado
    recorrente resetam em cada documento; sem ele, cada linha do batch é um documento.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.gdn_backend not in ("fla", "torch"):
            raise ValueError(f"gdn_backend desconhecido: {cfg.gdn_backend!r}")
        self.backend = cfg.gdn_backend
        self.qk_heads = cfg.gdn_qk_heads
        self.v_heads = cfg.gdn_v_heads
        self.head_dim = cfg.gdn_head_dim
        d = cfg.d_model
        qk_dim = self.qk_heads * self.head_dim
        v_dim = self.v_heads * self.head_dim

        self.q_proj = nn.Linear(d, qk_dim, bias=False)
        self.k_proj = nn.Linear(d, qk_dim, bias=False)
        self.v_proj = nn.Linear(d, v_dim, bias=False)
        self.g_proj = nn.Linear(d, v_dim, bias=False)
        self.a_proj = nn.Linear(d, self.v_heads, bias=False)
        self.b_proj = nn.Linear(d, self.v_heads, bias=False)

        # Conv depthwise [C,K] em fp32, com o init padrão do nn.Conv1d: U(±1/sqrt(K)).
        bound = cfg.conv_kernel**-0.5
        self.conv_q = nn.Parameter(torch.empty(qk_dim, cfg.conv_kernel).uniform_(-bound, bound))
        self.conv_k = nn.Parameter(torch.empty(qk_dim, cfg.conv_kernel).uniform_(-bound, bound))
        self.conv_v = nn.Parameter(torch.empty(v_dim, cfg.conv_kernel).uniform_(-bound, bound))

        # Decaimento: A ~ U(1, 16); dt log-uniforme em [1e-3, 1e-1] e dt_bias = softplus⁻¹(dt).
        self.A_log = nn.Parameter(torch.empty(self.v_heads).uniform_(1, 16).log())
        log_dt = torch.rand(self.v_heads) * (math.log(_DT_MAX) - math.log(_DT_MIN)) + math.log(_DT_MIN)
        dt = log_dt.exp()
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        self.o_norm = RMSNorm(self.head_dim, eps=cfg.norm_eps)
        self.o_proj = nn.Linear(v_dim, d, bias=False)

    def forward(self, x: Tensor, cu_seqlens: Tensor | None = None) -> Tensor:
        b, t, _ = x.shape
        doc_ids = None if cu_seqlens is None else doc_ids_from_cu_seqlens(cu_seqlens, t)

        q = self._conv_silu(self.q_proj(x), self.conv_q, doc_ids).view(b, t, self.qk_heads, self.head_dim)
        k = self._conv_silu(self.k_proj(x), self.conv_k, doc_ids).view(b, t, self.qk_heads, self.head_dim)
        v = self._conv_silu(self.v_proj(x), self.conv_v, doc_ids).view(b, t, self.v_heads, self.head_dim)
        rep = self.v_heads // self.qk_heads
        if rep > 1:  # GVA: cada head de q/k atende `rep` heads de v consecutivas
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)

        # g (log do decaimento, <= 0) sempre em fp32; beta pode ficar no dtype da ativação
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias.float())
        beta = self.b_proj(x).sigmoid()
        scale = self.head_dim**-0.5

        if self.backend == "fla":
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule

            o, _ = chunk_gated_delta_rule(
                q, k, v, g, beta, scale=scale, cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True
            )
        else:
            o = naive_gated_delta_rule(
                F.normalize(q, dim=-1), F.normalize(k, dim=-1), v, g, beta, scale=scale, cu_seqlens=cu_seqlens
            )

        gate = F.silu(self.g_proj(x)).view(b, t, self.v_heads, self.head_dim)
        return self.o_proj((self.o_norm(o) * gate).flatten(-2))

    @staticmethod
    def _conv_silu(h: Tensor, weight: Tensor, doc_ids: Tensor | None) -> Tensor:
        # peso fp32 convertido para o dtype da ativação: sob autocast bf16 a conv continua em bf16
        return F.silu(causal_conv1d_varlen(h, weight.to(h.dtype), doc_ids))
