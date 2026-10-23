"""Primitivas do modelo: RMSNorm e SwiGLU."""
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """`x * rsqrt(mean(x²) + eps) * weight`, calculado em float32 e devolvido no dtype da entrada."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(x.dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)[0]}, eps={self.eps}"


class SwiGLU(nn.Module):
    """FFN gated: `down_proj(silu(gate_proj(x)) * up_proj(x))`, sem bias."""

    def __init__(self, d_model: int, ffn_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
