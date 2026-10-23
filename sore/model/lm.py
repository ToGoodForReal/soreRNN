"""SoreLM: o modelo de linguagem soreQ4 completo.

`embed -> init_stream -> blocks (GDN/atenção 3:1) -> StreamFinal -> norm -> cabeça de LM`, com a
cabeça amarrada ao embedding (por padrão) e um módulo MTP de profundidade 1. Lotes empacotados
`[1, N]` com `cu_seqlens` não vazam entre documentos em nenhuma parte (tronco, loss e MTP).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from sore.model.attention import FLEX_MIN_LEN, GatedAttention, doc_causal_block_mask
from sore.model.block import Block
from sore.model.config import ModelConfig
from sore.model.gdn import GatedDeltaNet
from sore.model.layers import RMSNorm, SwiGLU
from sore.model.loss import ChunkCrossEntropy, chunked_lm_loss
from sore.model.mtp import MTPModule, mtp_labels, next_token_embeddings
from sore.model.residual import GatedHyperResidual, PlainResidual, StreamFinal, init_stream


@dataclass
class LMOutput:
    """`loss = lm_loss + mtp_weight * mtp_loss`, cada um a média sobre os seus tokens válidos.

    `n_tokens` é o número de tokens válidos da loss do LM. Sem `labels` tudo é `None` e `n_tokens = 0`.
    """

    loss: Tensor | None
    lm_loss: Tensor | None
    mtp_loss: Tensor | None
    n_tokens: int


class SoreLM(nn.Module):
    """LM híbrido soreQ4. `forward(input_ids [1,N], cu_seqlens, labels, inputs_embeds) -> LMOutput`.

    `labels[t]` é o alvo da posição t (já deslocado, `-100` = ignorar). O MTP usa
    `mtp_labels(labels)`, ou seja, o token t+2. Com `inputs_embeds` e sem `input_ids` o MTP é
    pulado, porque o embedding do próximo token vem de `input_ids`.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.mtp_layers > 1:
            raise NotImplementedError(f"mtp_layers={cfg.mtp_layers}: só há suporte a 0 ou 1 camada de MTP")
        if cfg.mtp_layers < 0:
            raise ValueError(f"mtp_layers deve ser >= 0; recebi {cfg.mtp_layers}")
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            Block(cfg, layer_idx=i, mixer_type=t) for i, t in enumerate(cfg.layer_types())
        )
        self.stream_final = StreamFinal(cfg)
        self.norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mtp = MTPModule(cfg) if cfg.mtp_layers > 0 else None
        self.lm_head = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.chunk_ce = ChunkCrossEntropy()  # sem parâmetros; o compile_model o compila in-place
        self.init_weights()

    @property
    def head_weight(self) -> Tensor:
        """Peso `[V, D]` da cabeça de LM (o do embedding, quando amarrado). O MTP usa o mesmo."""
        return self.embed.weight if self.lm_head is None else self.lm_head.weight

    @torch.no_grad()
    def init_weights(self) -> None:
        """normal(0, init_std) no embedding, na cabeça e nos `nn.Linear` de `blocks`/`mtp`.

        Ficam de fora os residuais (`res_mixer`, `res_ffn`) e o `stream_final`. Depois `o_proj` e
        `down_proj` são escalados por `1/sqrt(2 * n_layers)`. Os parâmetros especiais da GDN
        (`A_log`, `dt_bias`, `conv_*`) e os pesos das normas mantêm o próprio init, e o hyper-residual
        reaplica o seu (`reset_parameters`).
        """
        std = self.cfg.init_std
        self.embed.weight.normal_(0.0, std)
        if self.lm_head is not None:
            self.lm_head.weight.normal_(0.0, std)

        in_residual = {
            id(sub)
            for m in self.modules()
            if isinstance(m, (PlainResidual, GatedHyperResidual))
            for sub in m.modules()
        }
        for root in (self.blocks, self.mtp):
            if root is None:
                continue
            for m in root.modules():
                if isinstance(m, nn.Linear) and id(m) not in in_residual:
                    m.weight.normal_(0.0, std)

        out_scale = 1.0 / math.sqrt(2 * self.cfg.n_layers)
        for m in self.modules():
            if isinstance(m, (GatedDeltaNet, GatedAttention)):
                m.o_proj.weight.mul_(out_scale)
            elif isinstance(m, SwiGLU):
                m.down_proj.weight.mul_(out_scale)
            elif isinstance(m, GatedHyperResidual):
                m.reset_parameters()

    def attn_block_mask(self, x: Tensor, cu_seqlens: Tensor | None = None):
        """BlockMask do flex para `x [B,T,D]`, ou None (sdpa, CPU ou T < `FLEX_MIN_LEN`).

        Montado uma vez por forward e passado a todas as camadas de atenção, inclusive à do MTP
        (ruling R24). Sem ele cada camada montaria o seu, a cada forward e a cada recompute.
        """
        t = x.shape[1]
        if self.cfg.attn_backend != "flex" or not x.is_cuda or t < FLEX_MIN_LEN:
            return None
        return doc_causal_block_mask(cu_seqlens, t, x.device)

    def trunk(self, x: Tensor, cu_seqlens: Tensor | None = None, block_mask=None) -> Tensor:
        """Embeddings `[B,T,D]` -> hidden final `[B,T,D]` (blocos, `StreamFinal` e norma final).

        `block_mask` é o de `attn_block_mask` (montado aqui quando não vem pronto).
        """
        if block_mask is None:
            block_mask = self.attn_block_mask(x, cu_seqlens)
        H = init_stream(x, self.cfg)
        if cu_seqlens is not None:
            # O nº de documentos varia a cada micro-batch. Com o compile por Block (`compile_model`,
            # N estático) essa dimensão já entra dinâmica e não recompila a cada S. Em eager não faz nada.
            torch._dynamo.maybe_mark_dynamic(cu_seqlens, 0)
        use_checkpoint = self.cfg.grad_checkpoint and torch.is_grad_enabled()
        for block in self.blocks:
            if use_checkpoint:
                H = checkpoint(block, H, cu_seqlens, block_mask, use_reentrant=False)
            else:
                H = block(H, cu_seqlens, block_mask)
        return self.norm(self.stream_final(H))

    def forward(
        self,
        input_ids: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        labels: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> LMOutput:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("forward precisa de input_ids ou inputs_embeds")
        x = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        block_mask = self.attn_block_mask(x, cu_seqlens)  # um por forward: tronco e MTP
        h = self.trunk(x, cu_seqlens, block_mask)
        if labels is None:
            return LMOutput(loss=None, lm_loss=None, mtp_loss=None, n_tokens=0)

        weight = self.head_weight
        lm_sum, n_tokens = chunked_lm_loss(h.flatten(0, 1), weight, labels.flatten(), ce=self.chunk_ce)
        lm_loss = lm_sum / max(n_tokens, 1)
        loss, mtp_loss = lm_loss, None
        if self.mtp is not None and input_ids is not None:
            emb = x if inputs_embeds is None else self.embed(input_ids)
            h_mtp = self.mtp(h, next_token_embeddings(emb, cu_seqlens), cu_seqlens, block_mask)
            targets = mtp_labels(labels, cu_seqlens)
            mtp_sum, n_mtp = chunked_lm_loss(h_mtp.flatten(0, 1), weight, targets.flatten(), ce=self.chunk_ce)
            mtp_loss = mtp_sum / max(n_mtp, 1)
            loss = lm_loss + self.cfg.mtp_weight * mtp_loss
        return LMOutput(loss=loss, lm_loss=lm_loss, mtp_loss=mtp_loss, n_tokens=n_tokens)

    def logits(self, input_ids: Tensor, cu_seqlens: Tensor | None = None) -> Tensor:
        """Logits `[B,T,V]` do tronco (para testes e inferência; materializa `T x V`)."""
        return F.linear(self.trunk(self.embed(input_ids), cu_seqlens), self.head_weight)

    def num_params(self, exclude_embedding: bool = False) -> int:
        """Parâmetros únicos (o peso amarrado conta uma vez); `exclude_embedding` tira embedding e cabeça."""
        n = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n -= self.embed.weight.numel()
            if self.lm_head is not None:
                n -= self.lm_head.weight.numel()
        return n
