"""Módulo de dados do soreRNN: poda de tokenizador, filtros, deduplicação e pipeline streaming."""

from sore.data.prune_tokenizer import (
    build_dependency_graph,
    compute_ancestors,
    is_merge_closed,
)

__all__ = [
    "build_dependency_graph",
    "compute_ancestors",
    "is_merge_closed",
]
