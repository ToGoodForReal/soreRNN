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
_fla_patched = False

# `chunk_indices_without_sync` é adaptada de flash-linear-attention 0.5.2, `fla/ops/utils/index.py`
# (`_segmented_arange` e `prepare_chunk_indices`), licença MIT:
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# A diferença: os índices saem da GPU (o total de chunks vem do `cu_seqlens_cpu`), sem a cópia
# bloqueante final do original.


def chunk_indices_without_sync(
    cu_seqlens: Tensor, chunk_size: int, cu_seqlens_cpu: Tensor | None = None, *, original=None
) -> Tensor:
    """`fla.ops.utils.index.prepare_chunk_indices`, sem esperar a GPU: `[n_chunks, 2]` = (documento, chunk).

    O do FLA 0.5.2 calcula os índices na CPU a partir do `cu_seqlens_cpu` e depois faz
    `.to(cu_seqlens)`: uma cópia bloqueante de memória pageable, que sincroniza o stream da GPU
    uma vez por micro-batch. Aqui os índices saem da própria GPU: só o nº total de chunks (`n_chunks`,
    que dá o tamanho da saída) é lido, e vem do `cu_seqlens_cpu`; com `output_size` o
    `repeat_interleave` não lê a soma na GPU. Sem `cu_seqlens_cpu` delega ao `original` do FLA.
    """
    if cu_seqlens_cpu is None:
        return original(cu_seqlens, chunk_size)
    n_chunks = int(((cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1] + chunk_size - 1) // chunk_size).sum())
    counts = (cu_seqlens[1:] - cu_seqlens[:-1] + chunk_size - 1) // chunk_size
    seg_id = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device, dtype=counts.dtype), counts, output_size=n_chunks
    )
    seg_start = F.pad(counts.cumsum(0), (1, 0))[:-1]
    intra_chunk = torch.arange(n_chunks, device=counts.device, dtype=counts.dtype) - seg_start[seg_id]
    return torch.stack([seg_id, intra_chunk], 1)


def _patch_fla_chunk_indices() -> None:
    """Troca o `prepare_chunk_indices` do `chunk_gated_delta_rule` e da conv causal do FLA pelo `chunk_indices_without_sync`.

    Uma vez por processo. O resultado fica no `tensor_cache` do FLA (por identidade dos tensores),
    então as camadas GDN de um mesmo micro-batch, e o recompute do grad checkpoint, o calculam uma vez.
    A conv (`gdn_fla_conv`) chama o mesmo `prepare_chunk_indices(cu_seqlens, 64, cu_seqlens_cpu=...)`
    do módulo `fla.modules.conv.triton.ops` (3 vezes por camada, q/k/v): sem a troca cada chamada
    sincronizaria o stream; com a função cacheada compartilhada, o cálculo é um só por micro-batch.
    """
    global _fla_patched
    if _fla_patched:
        return
    import fla.modules.conv.triton.ops as fla_conv_ops
    import fla.ops.gated_delta_rule.chunk as fla_chunk
    from fla.utils import tensor_cache

    original = fla_chunk.prepare_chunk_indices

    @tensor_cache
    def prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=None):
        return chunk_indices_without_sync(cu_seqlens, chunk_size, cu_seqlens_cpu, original=original)

    fla_chunk.prepare_chunk_indices = prepare_chunk_indices
    fla_conv_ops.prepare_chunk_indices = prepare_chunk_indices
    _fla_patched = True


class GatedDeltaNet(nn.Module):
    """`o_proj(RMSNorm(o) * silu(g_proj(x)))`, com `o` vindo do gated delta rule sobre q/k/v pós-conv.

    No backend `fla` três opções do `ModelConfig` trocam trechos em PyTorch por kernels do FLA, sem mudar
    pesos nem resultado (`gdn_kernel_gate`: decaimento e sigmoid(beta) no kernel do delta rule;
    `gdn_fla_conv`: conv causal + SiLU do FLA; `fused_norm_gate`: RMSNorm·silu(gate) fundidos). O GVA é
    nativo no `fla` (q/k com `gdn_qk_heads`, sem `repeat_interleave`); o backend `torch` repete as heads.

    `forward(x [B,T,D], cu_seqlens)`: com `cu_seqlens` (`[S+1]`, exige `B == 1`) a conv e o estado
    recorrente resetam em cada documento; sem ele, cada linha do batch é um documento.
    `cu_seqlens_cpu` (mesmos valores, long, na CPU) vai ao kernel do FLA: sem ele o FLA lê o
    `cu_seqlens` da GPU no host (`prepare_chunk_indices`) e a GPU espera a CPU. Mesmo com ele, o
    FLA 0.5.2 copia os índices de chunk para a GPU com uma cópia bloqueante (sincroniza o stream);
    por isso o backend `fla` troca esse passo por `chunk_indices_without_sync`.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.gdn_backend not in ("fla", "torch"):
            raise ValueError(f"gdn_backend desconhecido: {cfg.gdn_backend!r}")
        self.backend = cfg.gdn_backend
        self.kernel_gate = cfg.gdn_kernel_gate
        self.fla_conv = cfg.gdn_fla_conv
        self.fused_norm_gate = cfg.fused_norm_gate
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

    def forward(self, x: Tensor, cu_seqlens: Tensor | None = None, cu_seqlens_cpu: Tensor | None = None) -> Tensor:
        b, t, _ = x.shape
        fla = self.backend == "fla"
        if fla:
            _patch_fla_chunk_indices()
        # a conv do FLA lê o `cu_seqlens` direto; a nossa precisa do documento de cada token
        doc_ids = None if cu_seqlens is None or self.fla_conv else doc_ids_from_cu_seqlens(cu_seqlens, t)

        def conv(h: Tensor, weight: Tensor) -> Tensor:
            return self._conv_silu(h, weight, doc_ids, cu_seqlens, cu_seqlens_cpu)

        q = conv(self.q_proj(x), self.conv_q).view(b, t, self.qk_heads, self.head_dim)
        k = conv(self.k_proj(x), self.conv_k).view(b, t, self.qk_heads, self.head_dim)
        v = conv(self.v_proj(x), self.conv_v).view(b, t, self.v_heads, self.head_dim)
        scale = self.head_dim**-0.5

        if fla:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule

            # GVA nativo (D4): o kernel recebe q/k com `qk_heads` e v com `v_heads`; a head i de v usa a
            # head i // (v_heads // qk_heads) de q/k, o mesmo agrupamento consecutivo do `repeat_interleave`
            # do backend torch (provado pelo teste de paridade). Sem a cópia de q/k, a l2norm e os
            # kernels de chunk leem `rep` vezes menos dados.
            if self.kernel_gate:
                # a_proj/b_proj crus: o kernel faz -exp(A_log)·softplus(a + dt_bias) (fp32) e sigmoid(b)
                gate_kwargs = dict(
                    use_gate_in_kernel=True, A_log=self.A_log, dt_bias=self.dt_bias, use_beta_sigmoid_in_kernel=True
                )
                g, beta = self.a_proj(x), self.b_proj(x)
            else:
                gate_kwargs = {}
                g, beta = self._gate_and_beta(x)
            o, _ = chunk_gated_delta_rule(
                q, k, v, g, beta, scale=scale, cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
                use_qk_l2norm_in_kernel=True, **gate_kwargs,
            )
        else:
            rep = self.v_heads // self.qk_heads
            if rep > 1:  # GVA: cada head de q/k atende `rep` heads de v consecutivas
                q = q.repeat_interleave(rep, dim=2)
                k = k.repeat_interleave(rep, dim=2)
            g, beta = self._gate_and_beta(x)
            o = naive_gated_delta_rule(
                F.normalize(q, dim=-1), F.normalize(k, dim=-1), v, g, beta, scale=scale, cu_seqlens=cu_seqlens
            )

        gate = self.g_proj(x).view(b, t, self.v_heads, self.head_dim)
        if self.fused_norm_gate:
            from fla.modules.fused_norm_gate import rms_norm_gated

            # RMSNorm(o) * silu(gate) num kernel só (cálculo em fp32, um único arredondamento para o dtype
            # de `o`); o funcional com `o_norm.weight` mantém o state_dict. Sem bias; eps do `o_norm`.
            o = rms_norm_gated(o, gate, self.o_norm.weight, None, activation="swish", eps=self.o_norm.eps)
        else:
            o = self.o_norm(o) * F.silu(gate)
        return self.o_proj(o.flatten(-2))

    def _gate_and_beta(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # g (log do decaimento, <= 0) sempre em fp32; beta pode ficar no dtype da ativação
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias.float())
        return g, self.b_proj(x).sigmoid()

    def _conv_silu(
        self, h: Tensor, weight: Tensor, doc_ids: Tensor | None, cu_seqlens: Tensor | None, cu_seqlens_cpu: Tensor | None
    ) -> Tensor:
        if self.fla_conv:
            from fla.modules.conv import causal_conv1d

            # Mesma orientação da nossa: o kernel aplica `weight[:, K-1]` ao token atual (tap i_w + W - 1,
            # i_w = 0) e zera o que vem antes do início de cada documento (`cu_seqlens`). O peso fp32 vai
            # direto (o kernel o lê em fp32), sem o cast para o dtype da ativação da conv nossa.
            return causal_conv1d(
                h, weight, activation="silu", cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu
            )[0]
        # peso fp32 convertido para o dtype da ativação: sob autocast bf16 a conv continua em bf16
        return F.silu(causal_conv1d_varlen(h, weight.to(h.dtype), doc_ids))
