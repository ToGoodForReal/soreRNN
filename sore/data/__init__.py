"""Módulo de dados do soreRNN: poda de tokenizador, filtros, deduplicação e pipeline streaming."""

from sore.data.dedup import (
    MinHashLSH,
    compute_minhash_signature,
)
from sore.data.filters import (
    check_alpha_density,
    check_length,
    check_line_repetition,
    check_ngram_repetition,
    check_pt_stopwords,
    clean_boilerplate,
    filter_document,
)
from sore.data.pipeline import (
    WeightedStreamIterator,
    run_pipeline,
)
from sore.data.prune_tokenizer import (
    build_dependency_graph,
    build_remapping_tensors,
    compute_ancestors,
    export_pruned_tokenizer,
    is_merge_closed,
    select_pruned_vocab,
)
from sore.data.sharder import BinarySharder

__all__ = [
    "BinarySharder",
    "MinHashLSH",
    "WeightedStreamIterator",
    "build_dependency_graph",
    "build_remapping_tensors",
    "check_alpha_density",
    "check_length",
    "check_line_repetition",
    "check_ngram_repetition",
    "check_pt_stopwords",
    "clean_boilerplate",
    "compute_ancestors",
    "compute_minhash_signature",
    "export_pruned_tokenizer",
    "filter_document",
    "is_merge_closed",
    "run_pipeline",
    "select_pruned_vocab",
]
