"""Módulo de deduplicação local em streaming via MinHash e Locality Sensitive Hashing (LSH).

Implementa assinatura MinHash determinística de 64 bits e anel circular deslizante
em memória para desduplicação de documentos em fluxo contínuo.
"""

from __future__ import annotations

import functools
import hashlib
from typing import Sequence

import numpy as np


@functools.lru_cache(maxsize=16)
def _get_permutations(num_perm: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Gera coeficientes determinísticos (a, b) para permutações universais uint64.

    Args:
        num_perm: Número de funções hash (permutações).
        seed: Semente pseudo-aleatória fixa para reprodutibilidade.

    Returns:
        Tupla com arrays a e b de shape (1, num_perm) e dtype uint64.
        O vetor `a` é forçado a ser ímpar para garantir invertibilidade em Z/(2^64).
    """
    rng = np.random.RandomState(seed)
    max_val = np.iinfo(np.int64).max
    a_raw = rng.randint(1, max_val, size=(1, num_perm), dtype=np.int64).astype(np.uint64)
    a = a_raw | np.uint64(1)  # Garante ímpar
    b = rng.randint(0, max_val, size=(1, num_perm), dtype=np.int64).astype(np.uint64)
    return a, b


def compute_minhash_signature(
    tokens: Sequence[str],
    num_perm: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """Calcula a assinatura MinHash 64-bit para uma sequência de tokens.

    Args:
        tokens: Sequência de tokens (ex: palavras, n-gramas).
        num_perm: Número de permutações (dimensão da assinatura). Padrão: 128.
        seed: Semente pseudo-aleatória fixa para reprodutibilidade. Padrão: 42.

    Returns:
        Array numpy 1D com dtype np.uint64 e tamanho `num_perm`.
        Se `tokens` estiver vazio, retorna um vetor de zeros sem erro.
    """
    if not tokens:
        return np.zeros(num_perm, dtype=np.uint64)

    a, b = _get_permutations(num_perm, seed)

    unique_tokens = list(set(tokens))
    # Hashing criptográfico blake2b de 64 bits para cada token distinto
    token_hashes = np.array(
        [
            int.from_bytes(hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest(), "little")
            for t in unique_tokens
        ],
        dtype=np.uint64,
    )[:, None]  # Shape: (N, 1)

    # Permutações universais vetorizadas em uint64: (X * a + b) mod 2^64
    permuted = token_hashes * a + b  # Shape: (N, num_perm)
    return np.min(permuted, axis=0)  # Shape: (num_perm,)


class MinHashLSH:
    """Indexador LSH (Locality Sensitive Hashing) com buffer circular deslizante em memória.

    Armazena até `capacity` assinaturas MinHash em um anel circular (sliding window),
    utilizando tabelas hash por bandas para busca de vizinhos próximos em tempo O(1),
    com verificação exata de similaridade de Jaccard estimada.
    """

    def __init__(
        self,
        capacity: int = 100000,
        num_perm: int = 128,
        threshold: float = 0.80,
        seed: int = 42,
        ngram_size: int = 5,
        r: int | None = None,
    ) -> None:
        """Inicializa o indexador MinHash LSH.

        Args:
            capacity: Capacidade máxima da janela deslizante (número de documentos em memória).
            num_perm: Número de permutações da assinatura MinHash. Padrão: 128.
            threshold: Limiar de similaridade de Jaccard mínima para considerar duplicata (ex: 0.80).
            seed: Semente determinística para as permutações.
            ngram_size: Tamanho de n-gramas de palavras para shingling de texto. Padrão: 5.
            r: Número de linhas por banda no LSH. Se None, usa min(4, num_perm).
        """
        if capacity <= 0:
            raise ValueError(f"capacity deve ser > 0, recebido: {capacity}")
        if num_perm <= 0:
            raise ValueError(f"num_perm deve ser > 0, recebido: {num_perm}")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold deve estar em [0.0, 1.0], recebido: {threshold}")

        self.capacity = capacity
        self.num_perm = num_perm
        self.threshold = threshold
        self.seed = seed
        self.ngram_size = ngram_size

        self.r = min(4, num_perm) if r is None else r
        if self.r <= 0 or self.r > num_perm:
            raise ValueError(f"r deve estar entre 1 e num_perm ({num_perm}), recebido: {self.r}")

        self.b = self.num_perm // self.r
        self._a, self._b_coef = _get_permutations(num_perm, seed)

        # Buffer circular de assinaturas: shape (capacity, num_perm)
        self._sig_buffer = np.zeros((capacity, num_perm), dtype=np.uint64)

        # Rastreia as chaves de banda associadas a cada slot para desalocação no eviction
        self._slot_bands: list[list[bytes] | None] = [None] * capacity

        # Tabelas de buckets por banda: self._tables[band_idx] -> dict[bytes, set[slot_index]]
        self._tables: list[dict[bytes, set[int]]] = [{} for _ in range(self.b)]

        self._head: int = 0
        self._size: int = 0

    def _tokenize(self, text: str) -> list[str]:
        """Extrai n-gramas de palavras de forma segura contra textos curtos ou vazios.

        Args:
            text: Texto do documento.

        Returns:
            Lista de n-gramas de palavras ou lista de palavras se len < ngram_size.
            Retorna lista vazia se o texto não contiver palavras.
        """
        words = text.split()
        if not words:
            return []
        if len(words) < self.ngram_size:
            return words
        return [" ".join(words[i : i + self.ngram_size]) for i in range(len(words) - self.ngram_size + 1)]

    def _extract_band_keys(self, sig: np.ndarray) -> list[bytes]:
        """Extrai as chaves binárias para cada uma das `b` bandas."""
        return [sig[i * self.r : (i + 1) * self.r].tobytes() for i in range(self.b)]

    def _insert_signature(self, sig: np.ndarray, band_keys: list[bytes]) -> None:
        """Insere uma nova assinatura na janela deslizante, descartando o item mais antigo se cheia."""
        idx = self._head

        # Se o slot já estiver ocupado (buffer cheio), desfaz o mapeamento do documento antigo
        if self._size == self.capacity:
            old_keys = self._slot_bands[idx]
            if old_keys is not None:
                for band_idx, old_key in enumerate(old_keys):
                    bucket = self._tables[band_idx].get(old_key)
                    if bucket is not None:
                        bucket.discard(idx)
                        if not bucket:
                            del self._tables[band_idx][old_key]

        # Grava a nova assinatura no slot
        self._sig_buffer[idx] = sig
        self._slot_bands[idx] = band_keys

        # Registra nos buckets das bandas
        for band_idx, key in enumerate(band_keys):
            bucket = self._tables[band_idx].setdefault(key, set())
            bucket.add(idx)

        # Avança o ponteiro circular
        self._head = (self._head + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def is_duplicate_and_insert(self, text: str) -> bool:
        """Verifica se o documento é duplicata de algum elemento na janela e o insere se não for.

        Documentos curtos (< 5 palavras) ou vazios são tratados de forma segura:
        - Textos vazios retornam False e não são indexados.
        - Textos curtos não dividem por zero e são indexados com base nas palavras existentes.

        Args:
            text: Conteúdo do documento em texto plano.

        Returns:
            True se for detectada similaridade estimada >= threshold (duplicata).
            False caso contrário (documento novo, que é então inserido no buffer).
        """
        tokens = self._tokenize(text)
        if not tokens:
            return False

        sig = compute_minhash_signature(tokens, num_perm=self.num_perm, seed=self.seed)
        band_keys = self._extract_band_keys(sig)

        # 1. Recupera candidatos colidindo em qualquer uma das bandas
        candidates: set[int] = set()
        for band_idx, key in enumerate(band_keys):
            bucket = self._tables[band_idx].get(key)
            if bucket:
                candidates.update(bucket)

        # 2. Verificação exata da similaridade de Jaccard estimada entre candidatos
        for cand_idx in candidates:
            cand_sig = self._sig_buffer[cand_idx]
            sim = float(np.count_nonzero(sig == cand_sig)) / float(self.num_perm)
            if sim >= self.threshold:
                return True

        # Documento único: insere na janela deslizante
        self._insert_signature(sig, band_keys)
        return False
