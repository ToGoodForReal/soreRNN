"""Block: uma camada do tronco, com dois sub-blocos pré-norma (mixer e SwiGLU), cada um com seu residual."""
from torch import Tensor, nn

from sore.model.attention import GatedAttention
from sore.model.config import ModelConfig
from sore.model.gdn import GatedDeltaNet
from sore.model.layers import RMSNorm, SwiGLU
from sore.model.residual import make_residual


class Block(nn.Module):
    """`H = res.write(H, f(norm(res.read(H))))` para o mixer e depois para a FFN.

    `mixer_type` é `"gdn"` (`GatedDeltaNet`) ou `"attn"` (`GatedAttention`). O stream `H` é
    `[B,T,D]` (residual plain) ou `[B,T,n,D]` (hyper). `cu_seqlens` só vai para o mixer e
    `block_mask` (o BlockMask do flex, montado uma vez por forward no `SoreLM`) só para a atenção.
    `cu_seqlens_cpu` (o mesmo `cu_seqlens`, na CPU) só vai para a GDN, que o repassa ao FLA para não
    sincronizar com a GPU.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int, mixer_type: str) -> None:
        super().__init__()
        if mixer_type == "gdn":
            mixer_cls = GatedDeltaNet
        elif mixer_type == "attn":
            mixer_cls = GatedAttention
        else:
            raise ValueError(f"mixer_type desconhecido: {mixer_type!r} (esperado 'gdn' ou 'attn')")
        self.mixer_type = mixer_type
        self.layer_idx = layer_idx

        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mixer = mixer_cls(cfg)
        self.res_mixer = make_residual(cfg, layer_idx, sub=0)
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_dim)
        self.res_ffn = make_residual(cfg, layer_idx, sub=1)

    def forward(
        self, H: Tensor, cu_seqlens: Tensor | None = None, block_mask=None, cu_seqlens_cpu: Tensor | None = None
    ) -> Tensor:
        x = self.norm1(self.res_mixer.read(H))
        if self.mixer_type == "attn":
            y = self.mixer(x, cu_seqlens, block_mask)
        else:
            y = self.mixer(x, cu_seqlens, cu_seqlens_cpu)
        H = self.res_mixer.write(H, y)
        H = self.res_ffn.write(H, self.ffn(self.norm2(self.res_ffn.read(H))))
        return H

    def extra_repr(self) -> str:
        return f"layer_idx={self.layer_idx}, mixer_type={self.mixer_type!r}"
