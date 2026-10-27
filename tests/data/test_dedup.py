"""Testes unitários para deduplicação MinHash LSH em streaming (sore.data.dedup)."""

import numpy as np
import pytest
from sore.data.dedup import MinHashLSH, compute_minhash_signature


def test_minhash_duplicate_detection():
    """Valida detecção de duplicatas próximas e aceitação de documentos distintos."""
    lsh = MinHashLSH(capacity=1000, num_perm=64, threshold=0.80)
    t1 = "O soreRNN é um modelo recorrente moderno treinado do zero em português com arquitetura baseada no Qwen4."
    t2 = "O soreRNN é um modelo recorrente moderno treinado do zero em português com arquitetura baseada no Qwen4!"  # Quase idêntico
    t3 = "A computação quântica utiliza q-bits superpostos para resolver problemas intratáveis em criptografia moderna."  # Totalmente diferente

    assert not lsh.is_duplicate_and_insert(t1)
    assert lsh.is_duplicate_and_insert(t2)  # Detecta como duplicata próxima
    assert not lsh.is_duplicate_and_insert(t3)  # Aceita texto diferente


def test_minhash_short_text_safety():
    """Garante que documentos curtos (< 5 palavras) ou vazios não causam crash ou divisão por zero."""
    lsh = MinHashLSH(capacity=100, num_perm=64)
    # Não quebra nem divide por zero com 1 ou 2 palavras ou string vazia
    assert not lsh.is_duplicate_and_insert("olá")
    assert not lsh.is_duplicate_and_insert("")
    assert not lsh.is_duplicate_and_insert("palavra única")


def test_minhash_window_rotation():
    """Valida a rotação da janela circular deslizante ao atingir a capacidade máxima."""
    capacity = 3
    lsh = MinHashLSH(capacity=capacity, num_perm=64, threshold=0.80)

    doc_a = "Primeiro documento longo para preencher a janela circular deslizante de testes com tokens distintos."
    doc_b = "Segundo documento longo completamente diferente do primeiro para ocupar o segundo slot da janela."
    doc_c = "Terceiro documento longo abordando outro assunto independente para preencher o terceiro slot do buffer."
    doc_d = "Quarto documento longo que deve provocar o descarte do documento mais antigo na fila circular."

    # Inserção inicial preenchendo a janela [A, B, C]
    assert not lsh.is_duplicate_and_insert(doc_a)
    assert not lsh.is_duplicate_and_insert(doc_b)
    assert not lsh.is_duplicate_and_insert(doc_c)

    # Inserção de D deve evictar A
    assert not lsh.is_duplicate_and_insert(doc_d)

    # doc_a foi descartado: reinserção não deve ser marcada como duplicata
    assert not lsh.is_duplicate_and_insert(doc_a)

    # doc_c ainda deve estar presente na janela
    assert lsh.is_duplicate_and_insert(doc_c)


def test_compute_minhash_signature_properties():
    """Valida formato, determinismo e robustez da assinatura MinHash."""
    tokens = ["aprendizado", "profundo", "redes", "recorrentes", "português", "modelo"]
    sig1 = compute_minhash_signature(tokens, num_perm=128)
    sig2 = compute_minhash_signature(tokens, num_perm=128)

    assert isinstance(sig1, np.ndarray)
    assert sig1.shape == (128,)
    assert sig1.dtype == np.uint64
    # Determinismo estrito
    np.testing.assert_array_equal(sig1, sig2)

    # Assinatura de lista vazia
    empty_sig = compute_minhash_signature([], num_perm=128)
    assert empty_sig.shape == (128,)
    assert empty_sig.dtype == np.uint64
    assert np.all(empty_sig == 0)


def test_minhash_exact_duplicate():
    """Valida que documentos idênticos são sempre detectados como duplicatas."""
    lsh = MinHashLSH(capacity=50, num_perm=64, threshold=0.80)
    text = "Texto idêntico que se repete múltiplas vezes no dataset para testar a retenção exata do filtro."
    assert not lsh.is_duplicate_and_insert(text)
    assert lsh.is_duplicate_and_insert(text)
