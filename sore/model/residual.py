"""Stream residual do tronco: `plain` (pré-norma padrão) ou `hyper` (gated hyper-connection).

Cada sub-bloco (mixer e FFN) tem seu próprio residual com o protocolo
`read(H) -> x [B,T,D]` e `write(H, y [B,T,D]) -> H`, e faz `H = res.write(H, f(norm(res.read(H))))`.

- `plain`: `H` é `[B,T,D]`, `read` é a identidade e `write` é `H + y`.
- `hyper`: `H` tem `n` ramos, `[B,T,n,D]`. Implementação própria (não há fórmula oficial),
  inspirada no "Gated Residual" do Qwen3.8-Flash-Next: gate de leitura low-rank, elemento a
  elemento e dependente dos dados, e gate de escrita escalar por ramo. Tudo é por posição (nada
  mistura no tempo), então documentos empacotados continuam isolados.

`init_stream` monta o stream a partir dos embeddings e `StreamFinal` o reduz de volta a `[B,T,D]`.
"""
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sore.model.config import ModelConfig


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _check_residual(cfg: ModelConfig) -> None:
    if cfg.residual not in ("plain", "hyper"):
        raise ValueError(f"residual desconhecido: {cfg.residual!r}")


def init_stream(x: Tensor, cfg: ModelConfig) -> Tensor:
    """Stream inicial: `x` (plain) ou `n` cópias de `x` em `[B,T,n,D]` (hyper)."""
    _check_residual(cfg)
    if cfg.residual == "plain":
        return x
    b, t, d = x.shape
    return x.unsqueeze(2).expand(b, t, cfg.hc_branches, d).clone()


class StreamFinal(nn.Module):
    """Reduz o stream a `[B,T,D]`: identidade (plain) ou soma ponderada por `softmax(mix_logits)` (hyper).

    `mix_logits` começa em zero, ou seja, a média uniforme dos ramos.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        _check_residual(cfg)
        self.hyper = cfg.residual == "hyper"
        if self.hyper:
            self.mix_logits = nn.Parameter(torch.zeros(cfg.hc_branches))

    def forward(self, H: Tensor) -> Tensor:
        if not self.hyper:
            return H
        return (torch.softmax(self.mix_logits, dim=0)[:, None] * H).sum(2)


class PlainResidual(nn.Module):
    """Residual pré-norma padrão: `read(H) = H`, `write(H, y) = H + y`."""

    def read(self, H: Tensor) -> Tensor:
        return H

    def write(self, H: Tensor, y: Tensor) -> Tensor:
        return H + y


class GatedHyperResidual(nn.Module):
    """Gated hyper-connection com `n` ramos; `H` tem shape `[B,T,n,D]`.

    Com `z = rms(H.mean(2))` (RMS norm sem peso):
    - `read`: `r = sigmoid(up(silu(down(z))))` em `[B,T,n,D]` e `x = (r * H).sum(2)`;
    - `write`: `w = 2 * sigmoid(write_bias + write_proj(z))` em `[B,T,n]` e `H + w[..., None] * y[:, :, None]`.

    Init: `up` com peso zero e bias `logit(1/n)` (leitura = média dos ramos); `write_proj` (W_w)
    zerado; `write_bias` dá `w = 1` no ramo `layer_idx % n` e `w = 0.1` nos outros (quebra de
    simetria). `z` é recalculado a partir do `H` recebido em `read` e em `write`, sem cache entre
    as chamadas, para o módulo continuar sem estado (seguro sob activation checkpointing).
    """

    def __init__(self, d: int, n: int, rank: int, layer_idx: int, eps: float = 1e-6) -> None:
        super().__init__()
        if n < 2:
            raise ValueError(f"hyper-connection precisa de pelo menos 2 ramos; recebi n={n}")
        self.d = d
        self.n = n
        self.layer_idx = layer_idx
        self.eps = eps
        self.down = nn.Linear(d, rank, bias=False)
        self.up = nn.Linear(rank, n * d, bias=True)
        self.write_proj = nn.Linear(d, n, bias=False)
        self.write_bias = nn.Parameter(torch.empty(n))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reaplica o init descrito na classe (`down` fica com o init padrão do `nn.Linear`)."""
        self.down.reset_parameters()
        self.up.weight.zero_()
        self.up.bias.fill_(_logit(1 / self.n))
        self.write_proj.weight.zero_()
        self.write_bias.fill_(_logit(0.05))  # 2 * sigmoid = 0.1
        self.write_bias[self.layer_idx % self.n] = 0.0  # 2 * sigmoid = 1

    def _z(self, H: Tensor) -> Tensor:
        return F.rms_norm(H.mean(2), (self.d,), eps=self.eps)

    def read(self, H: Tensor) -> Tensor:
        b, t, n, d = H.shape
        r = torch.sigmoid(self.up(F.silu(self.down(self._z(H))))).view(b, t, n, d)
        return (r * H).sum(2)

    def write(self, H: Tensor, y: Tensor) -> Tensor:
        w = 2 * torch.sigmoid(self.write_bias + self.write_proj(self._z(H)))
        return H + w[..., None] * y[:, :, None]

    def extra_repr(self) -> str:
        return f"d={self.d}, n={self.n}, layer_idx={self.layer_idx}, eps={self.eps}"


Residual = PlainResidual | GatedHyperResidual


def make_residual(cfg: ModelConfig, layer_idx: int, sub: int) -> Residual:
    """Residual de um sub-bloco (`sub`: 0 = mixer, 1 = FFN).

    `sub` ainda não muda nada: os dois sub-blocos da camada usam `layer_idx` no init assimétrico.
    """
    _check_residual(cfg)
    if cfg.residual == "plain":
        return PlainResidual()
    return GatedHyperResidual(cfg.d_model, cfg.hc_branches, cfg.hc_rank, layer_idx, eps=cfg.norm_eps)
