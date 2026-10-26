"""Subpacote de treino do soreRNN v2: otimizadores, schedule de LR, dados, checkpoint e trainer."""
from sore.train.checkpoint import load_checkpoint, save_checkpoint
from sore.train.config import Stage, TrainConfig, load_train_config, resolve_data_paths
from sore.train.data import Batch, PackedTokenStream, write_token_shard
from sore.train.fp8 import apply_fp8, fp8_supported
from sore.train.optim import build_optimizers, split_param_groups
from sore.train.schedule import wsd_multiplier
from sore.train.trainer import TrainSummary, train

__all__ = [
    "Batch",
    "PackedTokenStream",
    "Stage",
    "TrainConfig",
    "TrainSummary",
    "apply_fp8",
    "build_optimizers",
    "fp8_supported",
    "load_checkpoint",
    "load_train_config",
    "resolve_data_paths",
    "save_checkpoint",
    "split_param_groups",
    "train",
    "write_token_shard",
    "wsd_multiplier",
]
