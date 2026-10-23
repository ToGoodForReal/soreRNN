"""Subpacote do modelo soreRNN v2."""
from sore.model.attention import GatedAttention
from sore.model.block import Block
from sore.model.config import PRESETS, ModelConfig, get_preset
from sore.model.conv import causal_conv1d_varlen
from sore.model.gdn import GatedDeltaNet
from sore.model.gdn_ref import naive_gated_delta_rule
from sore.model.layers import RMSNorm, SwiGLU
from sore.model.lm import LMOutput, SoreLM
from sore.model.loss import chunked_lm_loss
from sore.model.mtp import MTPModule, mtp_labels
from sore.model.residual import StreamFinal, init_stream
from sore.model.varlen import doc_ids_from_cu_seqlens, positions_from_cu_seqlens

__all__ = [
    "PRESETS",
    "Block",
    "GatedAttention",
    "GatedDeltaNet",
    "LMOutput",
    "MTPModule",
    "ModelConfig",
    "RMSNorm",
    "SoreLM",
    "StreamFinal",
    "SwiGLU",
    "causal_conv1d_varlen",
    "chunked_lm_loss",
    "doc_ids_from_cu_seqlens",
    "get_preset",
    "init_stream",
    "mtp_labels",
    "naive_gated_delta_rule",
    "positions_from_cu_seqlens",
]
