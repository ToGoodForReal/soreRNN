"""Subpacote do modelo soreRNN v2."""
from sore.model.config import PRESETS, ModelConfig, get_preset
from sore.model.layers import RMSNorm, SwiGLU
from sore.model.varlen import doc_ids_from_cu_seqlens, positions_from_cu_seqlens

__all__ = [
    "PRESETS",
    "ModelConfig",
    "RMSNorm",
    "SwiGLU",
    "doc_ids_from_cu_seqlens",
    "get_preset",
    "positions_from_cu_seqlens",
]
