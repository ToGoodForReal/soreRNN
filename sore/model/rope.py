"""RoPE parcial: gira só as primeiras `int(head_dim * partial)` dims de cada head."""
import torch
from torch import Tensor, nn


class PartialRotary(nn.Module):
    """RoPE estilo GPT-NeoX (`rotate_half`) nas primeiras `rot_dim` dims; o resto passa intacto.

    Sem buffers: `inv_freq` é recalculado em fp32 a cada chamada (barato, `rot_dim / 2` elementos).
    Assim `.to(bfloat16)` no módulo nunca degrada as frequências e o `state_dict` fica limpo.
    """

    def __init__(self, head_dim: int, partial: float, theta: float) -> None:
        super().__init__()
        rot_dim = int(head_dim * partial)
        if rot_dim <= 0 or rot_dim % 2 != 0 or rot_dim > head_dim:
            raise ValueError(
                f"rot_dim=int({head_dim} * {partial})={rot_dim} deve ser par, positivo e <= head_dim"
            )
        self.head_dim = head_dim
        self.rot_dim = rot_dim
        self.theta = theta

    def forward(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        """`q [B,T,H,Dh]`, `k [B,T,Hkv,Dh]`, `positions [T]` (compartilhado pelo batch)."""
        cos, sin = self._cos_sin(positions, q.device, q.dtype)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    def _cos_sin(self, positions: Tensor, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        exponent = torch.arange(0, self.rot_dim, 2, device=device, dtype=torch.float32) / self.rot_dim
        inv_freq = 1.0 / self.theta**exponent
        freqs = positions.to(device=device, dtype=torch.float32)[:, None] * inv_freq[None, :]  # [T, rot/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, rot]
        # [1, T, 1, rot] para broadcast sobre batch e heads; fp32 até o cast final
        return emb.cos().to(dtype)[None, :, None, :], emb.sin().to(dtype)[None, :, None, :]

    def _rotate(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x_rot, x_pass = x[..., : self.rot_dim], x[..., self.rot_dim :]
        half = self.rot_dim // 2
        rotated = torch.cat((-x_rot[..., half:], x_rot[..., :half]), dim=-1)  # rotate_half
        return torch.cat((x_rot * cos + rotated * sin, x_pass), dim=-1)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, rot_dim={self.rot_dim}, theta={self.theta}"
