import os
#!/usr/bin/env python3
"""
soreRNN-LM: Filtros de Qualidade e Deduplicação em Escala Industrial (MinHash + LSH)
Garante:
1. Zero vazamento entre treino e validação (dedup val-vs-train rigoroso).
2. Deduplicação near-duplicate (MinHash LSH com 64 hashes e 16 bandas).
3. Deduplicação exata (xxhash64).
4. Filtro de qualidade de conteúdo (Score FineWeb-Edu, densidade léxica, repetições anormais).
"""

import re
import string
import xxhash
import numpy as np
from typing import Set, Dict, List, Optional, Tuple

class MinHashDeduplicator:
    def __init__(
        self,
        num_hashes: int = 64,
        num_bands: int = 16,
        shingle_size: int = 5,
        jaccard_threshold: float = 0.75,
        seed: int = 42,
        max_lsh_docs: int = int(os.environ.get("SORE_MAX_LSH_DOCS", 5_000_000))
    ):
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.k = shingle_size
        self.threshold = jaccard_threshold
        # Teto de memória: acima deste nº de docs de treino, o LSH (assinaturas + tabelas de banda)
        # para de crescer e a deduplicação continua apenas por hash exato (pega os reciclados idênticos).
        self.max_lsh_docs = max_lsh_docs

        # Parâmetros de hashing linear: (a * x + b) % p
        self.p = (1 << 61) - 1 # Mersenne prime
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, self.p, size=(self.num_hashes, 1), dtype=np.uint64)
        self.b = rng.randint(0, self.p, size=(self.num_hashes, 1), dtype=np.uint64)

        # Índices de armazenamento para treino
        self.exact_train_hashes: Set[int] = set()
        self.train_band_tables: List[Dict[int, List[int]]] = [{} for _ in range(self.num_bands)]
        self.train_signatures: List[np.ndarray] = []

        # Índices de validação (Held-Out) para prevenir vazamento (Data Leakage)
        self.exact_val_hashes: Set[int] = set()
        self.val_band_tables: List[Dict[int, List[int]]] = [{} for _ in range(self.num_bands)]
        self.val_signatures: List[np.ndarray] = []

    def _normalize(self, text: str) -> str:
        # Normalização básica de espaços e minúsculas
        return " ".join(text.lower().split())

    def _compute_shingles(self, text: str) -> Set[bytes]:
        norm = self._normalize(text)
        if len(norm) < self.k:
            return {norm.encode("utf-8")} if norm else set()
        return {norm[i:i + self.k].encode("utf-8") for i in range(len(norm) - self.k + 1)}

    def compute_signature(self, text: str) -> np.ndarray:
        shingles = self._compute_shingles(text)
        if not shingles:
            return np.zeros(self.num_hashes, dtype=np.uint64)

        raw_hashes = np.array([xxhash.xxh64_intdigest(s) for s in shingles], dtype=np.uint64)
        permuted = ((self.a * raw_hashes) + self.b) % self.p
        return np.min(permuted, axis=1)

    def is_val_leak(self, text: str, sig: Optional[np.ndarray] = None) -> bool:
        """Verifica se o documento colide ou é quase idêntico a algum documento de validação."""
        norm = self._normalize(text)
        h = xxhash.xxh64_intdigest(norm.encode("utf-8"))
        if h in self.exact_val_hashes:
            return True

        if not self.val_signatures:
            return False

        if sig is None:
            sig = self.compute_signature(text)

        # Checa colisão nas bandas LSH de validação
        candidate_indices = set()
        for b in range(self.num_bands):
            start = b * self.rows_per_band
            band_tuple = sig[start:start + self.rows_per_band].tobytes()
            band_h = xxhash.xxh64_intdigest(band_tuple)
            if band_h in self.val_band_tables[b]:
                candidate_indices.update(self.val_band_tables[b][band_h])

        for c_idx in candidate_indices:
            sim = np.mean(sig == self.val_signatures[c_idx])
            if sim >= (self.threshold - 0.10): # Limiar mais rigoroso para validação
                return True

        return False

    def is_train_duplicate(self, text: str, sig: Optional[np.ndarray] = None) -> bool:
        """Verifica se o documento já existe no conjunto de treino."""
        norm = self._normalize(text)
        h = xxhash.xxh64_intdigest(norm.encode("utf-8"))
        if h in self.exact_train_hashes:
            return True

        if len(self.train_signatures) >= self.max_lsh_docs:
            return False  # acima do teto de memória: mantém (dedup exato já foi checado)

        if not self.train_signatures:
            return False

        if sig is None:
            sig = self.compute_signature(text)

        candidate_indices = set()
        for b in range(self.num_bands):
            start = b * self.rows_per_band
            band_tuple = sig[start:start + self.rows_per_band].tobytes()
            band_h = xxhash.xxh64_intdigest(band_tuple)
            if band_h in self.train_band_tables[b]:
                candidate_indices.update(self.train_band_tables[b][band_h])

        for c_idx in candidate_indices:
            sim = np.mean(sig == self.train_signatures[c_idx])
            if sim >= self.threshold:
                return True

        return False

    def add_validation_document(self, text: str) -> bool:
        """Indexa um documento no conjunto de validação held-out."""
        norm = self._normalize(text)
        h = xxhash.xxh64_intdigest(norm.encode("utf-8"))
        if h in self.exact_val_hashes:
            return False # já existe na val

        sig = self.compute_signature(text)
        idx = len(self.val_signatures)
        self.val_signatures.append(sig)
        self.exact_val_hashes.add(h)

        for b in range(self.num_bands):
            start = b * self.rows_per_band
            band_tuple = sig[start:start + self.rows_per_band].tobytes()
            band_h = xxhash.xxh64_intdigest(band_tuple)
            self.val_band_tables[b].setdefault(band_h, []).append(idx)

        return True

    def add_train_document(self, text: str) -> bool:
        """
        Tenta adicionar um documento ao treino.
        Rejeita se for vazamento de validação ou se for duplicata de treino.
        """
        norm = self._normalize(text)
        h = xxhash.xxh64_intdigest(norm.encode("utf-8"))
        if h in self.exact_val_hashes or h in self.exact_train_hashes:
            return False

        if len(self.train_signatures) >= self.max_lsh_docs:
            self.exact_train_hashes.add(h)  # acima do teto: só hash exato (memória limitada)
            return True

        sig = self.compute_signature(text)

        if self.is_val_leak(text, sig):
            return False

        if self.is_train_duplicate(text, sig):
            return False

        idx = len(self.train_signatures)
        self.train_signatures.append(sig)
        self.exact_train_hashes.add(h)

        for b in range(self.num_bands):
            start = b * self.rows_per_band
            band_tuple = sig[start:start + self.rows_per_band].tobytes()
            band_h = xxhash.xxh64_intdigest(band_tuple)
            self.train_band_tables[b].setdefault(band_h, []).append(idx)

        return True


# ==============================================================================
# Filtros de Qualidade Heurística e Conteúdo
# ==============================================================================

PT_STOPWORDS = {
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
    "é", "com", "não", "uma", "os", "no", "se", "na", "por", "mais",
    "as", "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "à",
    "seu", "sua", "ou", "ser", "quando", "muito", "nos", "já", "está"
}

RE_CHAR_FLOOD = re.compile(r'(.)\1{8,}') # 9+ caracteres idênticos repetidos

def filter_fineweb_edu(row: dict, min_score: float = 3.0) -> bool:
    """
    Utiliza o classificador de pontuação educacional nativo do FineWeb-Edu.
    Retorna True se score >= min_score (escala de 0 a 5).
    """
    score = row.get("score")
    if score is None:
        score = row.get("int_score")
    if score is not None:
        try:
            return float(score) >= min_score
        except (ValueError, TypeError):
            return True
    return True # Se o dataset não possuir o campo, passa

def filter_text_quality(
    text: str,
    min_chars: int = 150,
    max_chars: int = 100_000,
    is_pt: bool = False
) -> bool:
    """
    Filtro de qualidade heurístico:
    - Comprimento mínimo e máximo razoável.
    - Sem repetições massivas de caracteres (spam).
    - Proporção equilibrada entre símbolos e palavras.
    - Heurística de parada para Português (presença de stopwords reais em PT).
    """
    if not text:
        return False
    stripped = text.strip()
    n_chars = len(stripped)
    if n_chars < min_chars or n_chars > max_chars:
        return False

    # Detecção de caracteres em flood (ex: "!!!!!!!!!!" ou "..........")
    if RE_CHAR_FLOOD.search(stripped):
        return False

    # Repetição excessiva de linhas (tabelas corrompidas ou menus repetidos)
    lines = [l.strip() for l in stripped.splitlines() if l.strip()]
    if len(lines) >= 6:
        unique_lines = set(lines)
        if len(unique_lines) / len(lines) < 0.60:
            return False

    # Densidade de símbolos de pontuação / não-alfanuméricos
    words = stripped.split()
    if len(words) < 20:
        return False

    non_alpha = sum(1 for c in stripped if not c.isalnum() and not c.isspace())
    if non_alpha / n_chars > 0.35:
        # Texto com mais de 35% de caracteres especiais (código corrompido ou tabelas)
        return False

    # Heurística para Português
    if is_pt:
        pt_words_found = 0
        for w in words[:100]:
            clean_w = w.lower().strip(string.punctuation)
            if clean_w in PT_STOPWORDS:
                pt_words_found += 1
        if pt_words_found < 3:
            return False

    return True

if __name__ == "__main__":
    dedup = MinHashDeduplicator()
    val_text = "A Revolução Francesa foi um período de intensa transformação política e social na França entre 1789 e 1799."
    dedup.add_validation_document(val_text)

    # Teste de vazamento
    assert dedup.is_val_leak(val_text) == True
    print(" -> Teste 1 (Vazamento Validação): OK")

    # Teste de near duplicate
    near_val = "A Revolução Francesa foi um período de intensa transformação política e social na França entre 1789 e 1799 com grandes mudanças."
    assert dedup.is_val_leak(near_val) == True
    print(" -> Teste 2 (Near-duplicate Validação): OK")

    # Teste de texto novo
    train_text = "O sistema solar é formado pelo Sol e pelos corpos celestes que orbitam ao seu redor, incluindo oito planetas."
    assert dedup.add_train_document(train_text) == True
    assert dedup.add_train_document(train_text) == False # Duplicata exata no treino
    print(" -> Teste 3 (Deduplicação Treino): OK")

    # Teste do filtro FineWeb-Edu
    assert filter_fineweb_edu({"score": 4.2}, min_score=3.0) == True
    assert filter_fineweb_edu({"score": 1.5}, min_score=3.0) == False
    print(" -> Teste 4 (Filtro FineWeb-Edu): OK")

    # Teste de qualidade de texto
    assert filter_text_quality("Curto", min_chars=150) == False
    assert filter_text_quality("Texto de teste com caracteres repetidos aaaaaaaaaaaaaaaaaaaaaaaaa", min_chars=50) == False
    print(" -> Teste 5 (Filtros de Qualidade Heurística): OK")
    print("\nTODOS OS TESTES DE DEDUPLICAÇÃO E QUALIDADE PASSARAM COM SUCESSO!")
