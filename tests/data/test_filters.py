"""Testes unitários para filtros heurísticos de qualidade e limpeza de texto (sore.data.filters)."""

import pytest
from sore.data.filters import (
    check_alpha_density,
    check_length,
    check_line_repetition,
    check_ngram_repetition,
    check_pt_stopwords,
    clean_boilerplate,
    filter_document,
)


def test_filter_valid_pt_document():
    """Valida se documento de alta qualidade em português é aprovado."""
    doc = (
        "O aprendizado de máquina moderno possibilita o desenvolvimento de modelos neurais "
        "avançados para o processamento de linguagem natural. Com o avanço das arquiteturas "
        "recorrentes modernas e da atenção linear, o treinamento de modelos na língua portuguesa "
        "atinge alta eficiência com excelente relação de tokens por segundo e qualidade semântica."
    )
    res = filter_document(doc, is_pt=True)
    assert res is not None
    assert isinstance(res, str)
    assert "aprendizado de máquina" in res


def test_filter_rejects_garbage_and_repetition():
    """Valida se lixo com baixa densidade de alfanuméricos e repetições são rejeitados."""
    # Pouco alfanumérico
    assert not check_alpha_density("%%% $$$$ @@@@ #### !!!! ???? **** ++++ ----")
    # Repetição excessiva de linhas (loop de SEO)
    repeated = "\n".join(["compre passagens aereas baratas aqui"] * 10)
    assert not check_line_repetition(repeated)
    assert filter_document(repeated, is_pt=True) is None


def test_filter_rejects_foreign_as_pt():
    """Valida se documento em idioma estrangeiro sem stopwords em português é rejeitado quando is_pt=True."""
    english_doc = (
        "Modern deep learning architectures allow rapid pretraining on consumer hardware. "
        "This is an English document that does not contain any Portuguese stopwords at all."
    )
    assert filter_document(english_doc, is_pt=True) is None


def test_check_length():
    """Testa limites de caracteres e palavras no filtro de comprimento."""
    assert not check_length("")
    assert not check_length("Texto curto.")

    # Texto com mais de 150 caracteres porém menos de 30 palavras
    long_words = "pneumoultramicroscopicossilicovulcanoconiose " * 5  # ~225 chars, 5 words
    assert not check_length(long_words, min_chars=150, min_words=30)

    # Texto com mais de 30 palavras porém menos de 150 caracteres
    short_words = "a b c d e f g h i j k l m n o p q r s t u v w x y z aa bb cc dd ee"  # 31 words, ~65 chars
    assert not check_length(short_words, min_chars=150, min_words=30)

    # Documento válido em tamanho
    valid = "Esta é uma sentença longa o suficiente para conter palavras reais e atingir todos os critérios mínimos estabelecidos para o pipeline de dados de pré-treino do modelo de linguagem soreRNN."
    assert len(valid) >= 150
    assert len(valid.split()) >= 25
    assert check_length(valid, min_chars=100, min_words=20)


def test_check_alpha_density():
    """Testa proporção mínima de caracteres alfanuméricos."""
    assert not check_alpha_density("")
    assert not check_alpha_density("12345 !@#$%^&*()_+=-~`{}[]|:;<>?,./")

    # Texto normal com letras, números e pontuação comum (proporção ~85%)
    normal_text = "O soreRNN utiliza 98304 tokens no vocabulário podado e treina com precisão bfloat16."
    assert check_alpha_density(normal_text, min_ratio=0.70)

    # Margem customizada
    symbols_text = "abc 123 !@#"  # 6 alfanuméricos de 11 caracteres totais (54.5%)
    assert not check_alpha_density(symbols_text, min_ratio=0.70)
    assert check_alpha_density(symbols_text, min_ratio=0.50)


def test_check_line_repetition():
    """Testa proporção máxima de linhas duplicadas."""
    # Menos de duas linhas válidas
    assert check_line_repetition("")
    assert check_line_repetition("Apenas uma linha única no documento.")

    # Linhas distintas com linhas em branco intercaladas
    distinct_lines = "Primeira linha informativa.\n\nSegunda linha sobre aprendizado.\n\nTerceira linha conclusiva."
    assert check_line_repetition(distinct_lines, max_ratio=0.30)

    # 4 linhas no total, sendo 2 duplicadas (2 duplicadas / 4 total = 50% > 30%)
    dups = "Linha repetida com frequência.\nLinha repetida com frequência.\nOutro conteúdo aqui.\nMais um conteúdo."
    assert not check_line_repetition(dups, max_ratio=0.30)
    assert check_line_repetition(dups, max_ratio=0.60)


def test_check_ngram_repetition():
    """Testa proporção de repetição de n-gramas (bloqueio de loops SEO)."""
    # Poucas palavras (menor que n=5)
    assert check_ngram_repetition("poucas palavras aqui", n=5)

    # Texto variado com 5-gramas únicos
    varied = (
        "redes neurais recorrentes utilizam estados ocultos para processar sequências "
        "com complexidade linear em relação ao tempo de inferência e treinamento"
    )
    assert check_ngram_repetition(varied, n=5, max_ratio=0.20)

    # Texto repetitivo em loop de 5-gramas
    loop = "compre passagens aereas baratas online " * 10
    assert not check_ngram_repetition(loop, n=5, max_ratio=0.20)


def test_check_pt_stopwords():
    """Testa identificação e contagem mínima de stopwords em português."""
    assert not check_pt_stopwords("")
    assert not check_pt_stopwords("algorithm model matrix gradient descent attention loss")

    # Documento com stopwords comuns do português (o, de, para, com, que, não)
    doc_pt = "O modelo de linguagem foi desenvolvido para processamento com redes que não requerem atenção quadrática."
    assert check_pt_stopwords(doc_pt, min_count=3)

    # Limite estrito de min_count
    poucas_stopwords = "Este computador funciona."  # "este"
    assert not check_pt_stopwords(poucas_stopwords, min_count=3)
    assert check_pt_stopwords(poucas_stopwords, min_count=1)


def test_clean_boilerplate():
    """Testa remoção de tags HTML, badges markdown, entidades e normalização de espaços."""
    raw_html = (
        "<div class='article-content'>\n"
        "  <h1>Título Principal</h1>\n"
        "  <script>var x = 10;</script>\n"
        "  <style>.body { color: red; }</style>\n"
        "  <!-- comentário html -->\n"
        "  <p>Texto do primeiro parágrafo com &amp; entidade HTML e &quot;aspas&quot;.</p>\n"
        "  <p>Segundo parágrafo com links <a href='https://example.com'>clique aqui</a>.</p>\n"
        "</div>"
    )
    cleaned = clean_boilerplate(raw_html)
    assert "<div" not in cleaned
    assert "<script" not in cleaned
    assert "var x = 10" not in cleaned
    assert "<style" not in cleaned
    assert "comentário html" not in cleaned
    assert "&amp;" not in cleaned
    assert '&quot;' not in cleaned
    assert "&" in cleaned
    assert '"aspas"' in cleaned
    assert "Título Principal" in cleaned
    assert "Texto do primeiro parágrafo" in cleaned
    assert "Segundo parágrafo" in cleaned

    # Remoção de badges markdown
    md_with_badges = (
        "# Repositório soreRNN\n\n"
        "[![CI](https://github.com/org/repo/workflows/CI/badge.svg)](https://github.com/org/repo/actions)\n"
        "![License](https://img.shields.io/badge/license-MIT-blue.svg)\n\n"
        "Descrição detalhada da arquitetura com múltiplos   espaços    consecutivos."
    )
    cleaned_md = clean_boilerplate(md_with_badges)
    assert "[![CI]" not in cleaned_md
    assert "![License]" not in cleaned_md
    assert "múltiplos espaços consecutivos" in cleaned_md


def test_filter_document_accepts_foreign_when_not_pt():
    """Valida se documentos estrangeiros de alta qualidade passam quando is_pt=False."""
    english_doc = (
        "Modern deep learning architectures allow rapid pretraining on consumer hardware. "
        "The mathematical formulation of state space models and linear attention mechanisms "
        "provides efficient sequence processing with constant memory footprint during token generation."
    )
    res = filter_document(english_doc, is_pt=False)
    assert res is not None
    assert "Modern deep learning architectures" in res


def test_filter_document_cleans_and_validates():
    """Valida se filter_document limpa boilerplate antes de retornar o texto aprovado."""
    doc_with_html = (
        "<div><p>O aprendizado de máquina moderno possibilita o desenvolvimento de modelos neurais "
        "avançados para o processamento de linguagem natural. Com o avanço das arquiteturas "
        "recorrentes modernas e da atenção linear, o treinamento de modelos na língua portuguesa "
        "atinge alta eficiência com excelente relação de tokens por segundo e qualidade semântica.</p></div>"
    )
    res = filter_document(doc_with_html, is_pt=True)
    assert res is not None
    assert "<div>" not in res
    assert "<p>" not in res
    assert res.startswith("O aprendizado de máquina")
