"""Subpacote do modelo soreRNN v2."""
from sore.model.config import PRESETS, ModelConfig, get_preset
from sore.model.conv import causal_conv1d_varlen
from sore.model.gdn_ref import naive_gated_delta_rule
from sore.model.layers import RMSNorm, SwiGLU
from sore.model.varlen import doc_ids_from_cu_seqlens, positions_from_cu_seqlens

__all__ = [
    "PRESETS",
    "ModelConfig",
    "RMSNorm",
    "SwiGLU",
    "causal_conv1d_varlen",
    "doc_ids_from_cu_seqlens",
    "get_preset",
    "naive_gated_delta_rule",
    "positions_from_cu_seqlens",
]
