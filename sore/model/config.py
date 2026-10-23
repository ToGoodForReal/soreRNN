"""Configuração do modelo (`ModelConfig`) e presets (`tiny`, `s150m`, `b0_8`)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Literal


@dataclass
class ModelConfig:
    # Dimensões
    vocab_size: int
    d_model: int
    n_layers: int
    ffn_dim: int

    # Layout: o padrão se repete n_layers / len(layer_pattern) vezes
    layer_pattern: tuple[str, ...] = ("gdn", "gdn", "gdn", "attn")

    # GatedDeltaNet
    gdn_qk_heads: int = 1
    gdn_v_heads: int = 1
    gdn_head_dim: int = 128
    conv_kernel: int = 4

    # Atenção gated
    attn_heads: int = 1
    attn_kv_heads: int = 1
    attn_head_dim: int = 128
    rope_theta: float = 1e7
    rope_partial: float = 0.25

    # Residual
    residual: Literal["plain", "hyper"] = "plain"
    hc_branches: int = 4
    hc_rank: int = 128

    # Multi-token prediction
    mtp_layers: int = 1
    mtp_weight: float = 0.3

    # Gerais
    tie_embeddings: bool = True
    norm_eps: float = 1e-6
    init_std: float = 0.02

    # Execução
    gdn_backend: Literal["fla", "torch"] = "fla"
    attn_backend: Literal["sdpa", "flex"] = "sdpa"
    grad_checkpoint: bool = False

    def __post_init__(self) -> None:
        # YAML/JSON devolvem lista; normaliza para tupla (hashable e comparável)
        self.layer_pattern = tuple(self.layer_pattern)
        if not self.layer_pattern:
            raise ValueError("layer_pattern não pode ser vazio")
        if self.n_layers % len(self.layer_pattern) != 0:
            raise ValueError(
                f"n_layers={self.n_layers} não é múltiplo de len(layer_pattern)={len(self.layer_pattern)}"
            )
        if self.gdn_v_heads % self.gdn_qk_heads != 0:
            raise ValueError(
                f"gdn_v_heads={self.gdn_v_heads} deve ser múltiplo de gdn_qk_heads={self.gdn_qk_heads}"
            )
        if self.attn_heads % self.attn_kv_heads != 0:
            raise ValueError(
                f"attn_heads={self.attn_heads} deve ser múltiplo de attn_kv_heads={self.attn_kv_heads}"
            )

    def layer_types(self) -> list[str]:
        """Tipo de cada camada (`"gdn"` ou `"attn"`), repetindo `layer_pattern`."""
        return list(self.layer_pattern) * (self.n_layers // len(self.layer_pattern))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"campos desconhecidos em ModelConfig: {sorted(unknown)}")
        return cls(**d)  # __post_init__ converte layer_pattern em tupla


PRESETS: dict[str, ModelConfig] = {
    "tiny": ModelConfig(
        vocab_size=512, d_model=128, n_layers=4, ffn_dim=384,
        gdn_qk_heads=1, gdn_v_heads=2, gdn_head_dim=64,
        attn_heads=2, attn_kv_heads=1, attn_head_dim=64,
        mtp_layers=1, gdn_backend="torch", attn_backend="sdpa",
    ),
    "s150m": ModelConfig(
        vocab_size=65536, d_model=768, n_layers=12, ffn_dim=2048,
        gdn_qk_heads=6, gdn_v_heads=12,
        attn_heads=6, attn_kv_heads=2,
        mtp_layers=1, gdn_backend="fla", attn_backend="flex",
    ),
    "b0_8": ModelConfig(
        vocab_size=65536, d_model=1280, n_layers=24, ffn_dim=4352,
        gdn_qk_heads=10, gdn_v_heads=20,
        attn_heads=10, attn_kv_heads=2,
        mtp_layers=1, gdn_backend="fla", attn_backend="flex",
    ),
}


def get_preset(name: str, **overrides: Any) -> ModelConfig:
    """Devolve uma cópia do preset `name` com `overrides` aplicados (PRESETS nunca é mutado)."""
    if name not in PRESETS:
        raise KeyError(f"preset desconhecido {name!r}; disponíveis: {sorted(PRESETS)}")
    return replace(PRESETS[name], **overrides)
