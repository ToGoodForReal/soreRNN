"""Orquestrador do pipeline de dados de pré-treino em streaming e particionamento de shards.

Integra multiplexação probabilística de fluxos de documentos, filtros heurísticos de qualidade,
deduplicação local via MinHash LSH, tokenização contínua e gravação atômica em shards binários uint32.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Iterator

from sore.data.dedup import MinHashLSH
from sore.data.filters import (
    check_alpha_density,
    check_length,
    check_line_repetition,
    check_ngram_repetition,
    check_pt_stopwords,
    clean_boilerplate,
)
from sore.data.sharder import BinarySharder

logger = logging.getLogger(__name__)


class WeightedStreamIterator:
    """Iterador que consome múltiplos fluxos de documentos com amostragem estocástica ponderada.

    A cada invocação de `next()`, sorteia uma fonte com base nos pesos normalizados.
    Se uma fonte for esgotada (`StopIteration`), ela é automaticamente removida do conjunto
    de fontes ativas e os pesos remanescentes são renormalizados. O iterador só encerra
    quando todas as fontes forem completamente exauridas.
    """

    def __init__(
        self,
        sources: dict[str, Any],
        weights: dict[str, float] | None = None,
        seed: int = 42,
    ) -> None:
        """Inicializa o multiplexador ponderado de fluxos.

        Args:
            sources: Dicionário mapeando nome da fonte -> iterável ou iterador de documentos.
            weights: Dicionário mapeando nome da fonte -> peso relativo positivo.
                     Se None, atribui peso uniforme (1.0 / N) para todas as fontes.
            seed: Semente determinística para o gerador pseudo-aleatório.
        """
        if not sources:
            raise ValueError("sources cannot be empty.")

        self._active_sources: dict[str, Iterator[Any]] = {
            name: iter(src) for name, src in sources.items()
        }

        if weights is None:
            self._weights: dict[str, float] = {
                name: 1.0 / len(sources) for name in sources
            }
        else:
            for name, w in weights.items():
                if w <= 0:
                    raise ValueError(f"Weight for '{name}' must be positive, got {w}")
            self._weights = dict(weights)

        # Garante que todas as fontes em sources tenham peso definido
        for name in self._active_sources:
            if name not in self._weights:
                raise ValueError(f"Missing weight for source: '{name}'")

        self._rng = random.Random(seed)
        self.last_source: str | None = None

    def __iter__(self) -> WeightedStreamIterator:
        return self

    def __next__(self) -> Any:
        while self._active_sources:
            keys = list(self._active_sources.keys())
            weights = [self._weights[k] for k in keys]
            total_w = sum(weights)

            if total_w <= 0:
                break

            chosen_source = self._rng.choices(keys, weights=weights, k=1)[0]
            iterator = self._active_sources[chosen_source]

            try:
                item = next(iterator)
                self.last_source = chosen_source

                if isinstance(item, dict):
                    item = dict(item)
                    if "_source" not in item:
                        item["_source"] = chosen_source

                return item
            except StopIteration:
                # Fonte esgotada: remove das fontes ativas e continua sorteando entre as restantes
                del self._active_sources[chosen_source]

        raise StopIteration


def run_pipeline(
    sources: dict[str, Any],
    weights: dict[str, float] | None = None,
    tokenizer: Any = None,
    out_dir: str | Path = "data/shards",
    max_tokens: int = 268435456,
    shard_size: int = 268435456,
    unit_len: int | None = None,
    eos_id: int = 98303,
    lsh_capacity: int = 100000,
    lsh_threshold: float = 0.80,
    min_chars: int = 150,
    min_words: int = 30,
    min_alpha_ratio: float = 0.70,
    max_line_dup_ratio: float = 0.30,
    max_ngram_dup_ratio: float = 0.20,
    ngram_size: int = 5,
    min_pt_stopwords: int = 3,
    pt_sources: set[str] | None = None,
    seed: int = 42,
    show_progress: bool = True,
    return_stats: bool = False,
) -> list[Path] | tuple[list[Path], dict[str, Any]]:
    """Executa a esteira ponta a ponta de processamento de dados para pré-treinamento.

    Conecta:
    1. WeightedStreamIterator: amostragem probabilística multiplexada;
    2. clean_boilerplate & filtros: regras heurísticas com detalhamento de causas de rejeição;
    3. MinHashLSH: deduplicação local em streaming;
    4. Tokenizer: conversão de texto higienizado em token IDs uint32;
    5. BinarySharder: empacotamento contíguo em blocos uint32 com sidecar JSON determinístico.

    Args:
        sources: Dicionário nome -> iterador/iterável de documentos.
        weights: Pesos das fontes. Padrão uniforme.
        tokenizer: Tokenizador compatível (HuggingFace Tokenizer, FastTokenizer ou mock com .encode).
        out_dir: Diretório de saída dos shards binários e sidecars.
        max_tokens: Quantidade total máxima de tokens a serem gerados.
        shard_size: Capacidade máxima de tokens por shard binário. Padrão: 2^28 = 268.435.456.
        unit_len: Tamanho do bloco de empacotamento em tokens. Se None, usa 8192 (se shard_size múltiplo) ou 1.
        eos_id: ID do token especial EOS/EOD. Padrão: 98303.
        lsh_capacity: Capacidade da janela deslizante do MinHash LSH. Padrão: 100.000 docs.
        lsh_threshold: Limiar de similaridade de Jaccard para considerar duplicata. Padrão: 0.80.
        min_chars: Comprimento mínimo em caracteres para filtros de qualidade.
        min_words: Mínimo de palavras no documento.
        min_alpha_ratio: Densidade alfanumérica mínima.
        max_line_dup_ratio: Limite de proporção de linhas duplicadas.
        max_ngram_dup_ratio: Limite de proporção de repetição de n-gramas.
        ngram_size: Ordem n do n-grama para teste de repetição.
        min_pt_stopwords: Mínimo de stopwords em português exigidas.
        pt_sources: Conjunto explícito de fontes em português. Se None, detecta automaticamente.
        seed: Semente determinística para geradores pseudo-aleatórios.
        show_progress: Se True e tqdm instalado, exibe barra de progresso.
        return_stats: Se True, retorna tupla (shards, stats); se False, retorna apenas shards.

    Returns:
        Lista de caminhos dos shards binários (.bin) gerados, ou (shards, stats) se return_stats=True.
    """
    if tokenizer is None:
        raise ValueError("tokenizer must be provided to run_pipeline.")

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    if unit_len is None:
        unit_len = 8192 if (shard_size % 8192 == 0) else 1

    lsh = MinHashLSH(
        capacity=lsh_capacity,
        threshold=lsh_threshold,
        seed=seed,
    )

    stream = WeightedStreamIterator(
        sources=sources,
        weights=weights,
        seed=seed,
    )

    stats: dict[str, Any] = {
        "documents_seen": 0,
        "documents_filtered": 0,
        "documents_duplicated": 0,
        "documents_accepted": 0,
        "tokens_written": 0,
        "filter_rejections": {
            "length": 0,
            "alpha_density": 0,
            "line_repetition": 0,
            "ngram_repetition": 0,
            "pt_stopwords": 0,
        },
    }

    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=max_tokens, unit="tok", desc="Tokens processados")
        except ImportError:
            pbar = None

    shards: list[Path] = []
    with BinarySharder(
        out_dir=out_dir_path,
        shard_size=shard_size,
        eos_id=eos_id,
        unit_len=unit_len,
    ) as sharder:
        try:
            for doc in stream:
                stats["documents_seen"] += 1

                # Extração de texto com fallbacks de coluna ("text", "content", "raw_content")
                if isinstance(doc, dict):
                    text = doc.get("text") or doc.get("content") or doc.get("raw_content") or ""
                    source_name = doc.get("_source", doc.get("source", "unknown"))
                    if "is_pt" in doc:
                        is_pt = bool(doc["is_pt"])
                    elif pt_sources is not None:
                        is_pt = source_name in pt_sources
                    else:
                        s_lower = str(source_name).lower()
                        if (
                            s_lower.startswith(("pt_", "pt-", "por_"))
                            or s_lower == "pt"
                            or "_pt_" in s_lower
                            or s_lower.endswith(("_pt", "-pt"))
                        ):
                            is_pt = True
                        elif "en" in s_lower or "code" in s_lower or "math" in s_lower or "python" in s_lower:
                            is_pt = False
                        else:
                            is_pt = True
                elif isinstance(doc, str):
                    text = doc
                    source_name = getattr(stream, "last_source", "unknown")
                    is_pt = True
                else:
                    text = str(doc)
                    source_name = getattr(stream, "last_source", "unknown")
                    is_pt = True

                if not text:
                    stats["documents_filtered"] += 1
                    stats["filter_rejections"]["length"] += 1
                    continue

                # 1. Filtros Heurísticos de Qualidade com rastreamento detalhado de rejeições
                cleaned_text = clean_boilerplate(text)

                rejection_reason = None
                if not check_length(cleaned_text, min_chars=min_chars, min_words=min_words):
                    rejection_reason = "length"
                elif not check_alpha_density(cleaned_text, min_ratio=min_alpha_ratio):
                    rejection_reason = "alpha_density"
                elif not check_line_repetition(cleaned_text, max_ratio=max_line_dup_ratio):
                    rejection_reason = "line_repetition"
                elif not check_ngram_repetition(cleaned_text, n=ngram_size, max_ratio=max_ngram_dup_ratio):
                    rejection_reason = "ngram_repetition"
                elif is_pt and not check_pt_stopwords(cleaned_text, min_count=min_pt_stopwords):
                    rejection_reason = "pt_stopwords"

                if rejection_reason is not None:
                    stats["documents_filtered"] += 1
                    stats["filter_rejections"][rejection_reason] += 1
                    continue

                # 2. Deduplicação Local em Janela Deslizante (MinHash LSH)
                if lsh.is_duplicate_and_insert(cleaned_text):
                    stats["documents_duplicated"] += 1
                    continue

                # 3. Tokenização
                if hasattr(tokenizer, "encode"):
                    encoded = tokenizer.encode(cleaned_text)
                elif callable(tokenizer):
                    encoded = tokenizer(cleaned_text)
                else:
                    raise TypeError(f"Unsupported tokenizer type: {type(tokenizer)}")

                if hasattr(encoded, "ids"):
                    token_ids = encoded.ids
                elif isinstance(encoded, dict) and "input_ids" in encoded:
                    token_ids = encoded["input_ids"]
                else:
                    token_ids = encoded

                if not token_ids:
                    continue

                # 4. Verificação de Limite de Tokens e Gravação em Shards
                remaining_tokens = max_tokens - stats["tokens_written"]
                if remaining_tokens <= 0:
                    break

                doc_len = len(token_ids) + 1  # tokens do documento + eos_id
                if doc_len > remaining_tokens:
                    take_tokens = max(0, remaining_tokens - 1)
                    token_ids = token_ids[:take_tokens]
                    doc_tokens_to_add = take_tokens + 1
                else:
                    doc_tokens_to_add = doc_len

                sharder.add_document(token_ids, source_name=source_name)
                stats["documents_accepted"] += 1
                stats["tokens_written"] += doc_tokens_to_add

                if pbar is not None:
                    pbar.update(doc_tokens_to_add)

                if stats["tokens_written"] >= max_tokens:
                    break

        finally:
            if pbar is not None:
                pbar.close()

    shards = sharder.written_shards

    if return_stats:
        return shards, stats
    return shards
