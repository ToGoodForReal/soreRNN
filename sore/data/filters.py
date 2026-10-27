"""Filtros heurísticos de qualidade e higienização de texto para o pipeline de dados.

Implementa regras de limpeza de boilerplate, verificação de densidade alfanumérica,
limites de comprimento em caracteres e palavras, detecção de loops de repetição
(linhas e n-gramas) e verificação de stopwords da língua portuguesa.
"""

from __future__ import annotations

import html
import re
from collections import Counter

# Stopwords de alta frequência da língua portuguesa para validação de idioma
PT_STOPWORDS: frozenset[str] = frozenset({
    # Artigos e contrações comuns
    "o", "os", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas",
    "num", "numa", "nuns", "numas",
    "ao", "aos", "à", "às",
    "pelo", "pela", "pelos", "pelas",
    # Preposições
    "por", "para", "com", "sem", "sob", "sobre", "entre", "até", "perante", "trás", "contra",
    # Conjunções e conectivos
    "que", "e", "ou", "mas", "porém", "contudo", "todavia", "portanto", "entretanto",
    "porque", "porquê", "pois", "como", "quando", "onde", "quanto", "se",
    # Pronomes
    "eu", "tu", "ele", "ela", "nós", "vós", "eles", "elas", "você", "vocês",
    "meu", "minha", "meus", "minhas",
    "teu", "tua", "teus", "tuas",
    "seu", "sua", "seus", "suas",
    "nosso", "nossa", "nossos", "nossas",
    "dele", "dela", "deles", "delas",
    "este", "esta", "estes", "estas", "isto",
    "esse", "essa", "esses", "essas", "isso",
    "aquele", "aquela", "aqueles", "aquelas", "aquilo",
    "quem", "qual", "quais", "cujo", "cuja", "cujos", "cujas",
    # Advérbios e partículas
    "não", "mais", "muito", "já", "também", "só", "mesmo", "ainda", "depois",
    "antes", "sempre", "nunca", "talvez", "assim", "bem", "menos", "tão",
    # Formas verbais auxiliares frequentes
    "é", "são", "era", "eram", "foi", "foram", "ser", "seja", "sejam", "sendo", "sido",
    "tem", "têm", "tinha", "tinham", "teve", "ter", "tenha", "tenham", "tendo", "tido",
    "está", "estão", "estava", "estavam", "esteve", "estar", "esteja", "estejam", "estando", "estado",
    "havia", "houve", "haver", "haja", "há",
})

# Padrões compilados para higienização de boilerplate
_RE_SCRIPT_STYLE = re.compile(r"<(script|style)[\s\S]*?</\1>", re.IGNORECASE)
_RE_HTML_COMMENTS = re.compile(r"<!--[\s\S]*?-->")
_RE_HTML_TAGS = re.compile(r"<[^>]+>")
_RE_MD_BADGE_LINK = re.compile(r"\[!\[.*?\]\(.*?\)\]\(.*?\)")
_RE_MD_IMAGE_BADGE = re.compile(r"!\[.*?\]\(.*?\)")
_RE_MULTIPLE_HSPACES = re.compile(r"[^\S\r\n]+")
_RE_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
_RE_WORD_TOKENS = re.compile(r"\b\w+\b")


def check_length(text: str, min_chars: int = 150, min_words: int = 30) -> bool:
    """Verifica se o texto satisfaz os critérios de comprimento mínimo em caracteres e palavras.

    Args:
        text: Texto a avaliar.
        min_chars: Quantidade mínima de caracteres totais. Padrão: 150.
        min_words: Quantidade mínima de palavras separadas por espaço. Padrão: 30.

    Returns:
        True se o texto tiver pelo menos min_chars caracteres e min_words palavras.
    """
    if not text:
        return False
    if len(text) < min_chars:
        return False
    words = text.split()
    return len(words) >= min_words


def check_alpha_density(text: str, min_ratio: float = 0.70) -> bool:
    """Verifica se a densidade alfanumérica do documento atinge a razão mínima.

    Descarta sequências binárias, dumps de memória e ruído de pontuação excessiva.

    Args:
        text: Texto a avaliar.
        min_ratio: Proporção mínima de caracteres alfanuméricos em relação ao comprimento total.

    Returns:
        True se a razão alfanumérica for maior ou igual a min_ratio.
    """
    if not text:
        return False
    alpha_count = sum(1 for c in text if c.isalnum())
    return (alpha_count / len(text)) >= min_ratio


def check_line_repetition(text: str, max_ratio: float = 0.30) -> bool:
    """Verifica se a proporção de linhas duplicadas não excede o limiar tolerado.

    Linhas em branco são desconsideradas no cálculo da proporção.

    Args:
        text: Texto a avaliar.
        max_ratio: Proporção máxima de linhas duplicadas permitida. Padrão: 0.30.

    Returns:
        True se a proporção de linhas duplicadas for menor ou igual a max_ratio.
    """
    if not text:
        return True
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return True
    line_counts = Counter(lines)
    duplicate_lines_count = sum(count for count in line_counts.values() if count > 1)
    duplicate_ratio = duplicate_lines_count / len(lines)
    return duplicate_ratio <= max_ratio


def check_ngram_repetition(text: str, n: int = 5, max_ratio: float = 0.20) -> bool:
    """Verifica se a proporção de n-gramas repetidos não excede o limiar especificado.

    Detecta e bloqueia spams de palavras-chave e loops automáticos de SEO.

    Args:
        text: Texto a avaliar.
        n: Tamanho do n-grama em palavras. Padrão: 5.
        max_ratio: Proporção máxima de n-gramas repetidos. Padrão: 0.20.

    Returns:
        True se a proporção de n-gramas repetidos for menor ou igual a max_ratio.
    """
    if not text or n <= 0:
        return True
    words = _RE_WORD_TOKENS.findall(text.lower())
    if len(words) < n:
        return True
    total_ngrams = len(words) - n + 1
    unique_ngrams = len({tuple(words[i : i + n]) for i in range(total_ngrams)})
    duplicate_ratio = (total_ngrams - unique_ngrams) / total_ngrams
    return duplicate_ratio <= max_ratio


def check_pt_stopwords(text: str, min_count: int = 3) -> bool:
    """Verifica se o texto contém uma contagem mínima de stopwords em português.

    Utilizado para descartar textos em língua estrangeira em fontes identificadas como PT.

    Args:
        text: Texto a avaliar.
        min_count: Número mínimo de ocorrências de stopwords necessárias. Padrão: 3.

    Returns:
        True se a contagem total de stopwords em português for >= min_count.
    """
    if not text:
        return False
    words = _RE_WORD_TOKENS.findall(text.lower())
    count = sum(1 for w in words if w in PT_STOPWORDS)
    return count >= min_count


def clean_boilerplate(text: str) -> str:
    """Higieniza o texto removendo tags HTML, scripts, estilos, badges e espaços redundantes.

    Args:
        text: Texto original com possíveis resíduos web ou markdown.

    Returns:
        Texto limpo e normalizado.
    """
    if not text:
        return ""

    # Remove blocos de script e style por completo
    cleaned = _RE_SCRIPT_STYLE.sub("", text)

    # Remove comentários HTML
    cleaned = _RE_HTML_COMMENTS.sub("", cleaned)

    # Substitui tags HTML por espaço para evitar aglutinação de palavras
    cleaned = _RE_HTML_TAGS.sub(" ", cleaned)

    # Decodifica entidades HTML como &amp;, &quot;, &lt;, &gt;
    cleaned = html.unescape(cleaned)

    # Normaliza non-breaking spaces
    cleaned = cleaned.replace("\xa0", " ")

    # Remove badges markdown (com ou sem link circundante)
    cleaned = _RE_MD_BADGE_LINK.sub("", cleaned)
    cleaned = _RE_MD_IMAGE_BADGE.sub("", cleaned)

    # Normaliza espaços horizontais linha a linha
    lines = [_RE_MULTIPLE_HSPACES.sub(" ", line).strip() for line in cleaned.splitlines()]
    cleaned = "\n".join(lines)

    # Colapsa mais de duas quebras de linha consecutivas
    cleaned = _RE_MULTIPLE_NEWLINES.sub("\n\n", cleaned)

    return cleaned.strip()


def filter_document(
    text: str,
    is_pt: bool = True,
    min_chars: int = 150,
    min_words: int = 30,
    min_alpha_ratio: float = 0.70,
    max_line_dup_ratio: float = 0.30,
    max_ngram_dup_ratio: float = 0.20,
    ngram_size: int = 5,
    min_pt_stopwords: int = 3,
) -> str | None:
    """Aplica a esteira completa de filtros heurísticos e higienização em um documento.

    Executa sequencialmente:
    1. Higienização de boilerplate (tags HTML, badges, normalização de espaços);
    2. Comprimento mínimo em caracteres e palavras;
    3. Densidade alfanumérica mínima;
    4. Limite de proporção de repetição de linhas;
    5. Limite de proporção de repetição de n-gramas;
    6. Verificação de stopwords em português (se is_pt=True).

    Args:
        text: Documento de texto bruto.
        is_pt: Se True, exige quantidade mínima de stopwords em português.
        min_chars: Quantidade mínima de caracteres.
        min_words: Quantidade mínima de palavras.
        min_alpha_ratio: Razão alfanumérica mínima.
        max_line_dup_ratio: Proporção máxima de linhas duplicadas.
        max_ngram_dup_ratio: Proporção máxima de n-gramas repetidos.
        ngram_size: Ordem n do n-grama de palavras.
        min_pt_stopwords: Mínimo de ocorrências de stopwords PT quando is_pt=True.

    Returns:
        O texto higienizado se satisfizer todas as regras; caso contrário, None.
    """
    if not text:
        return None

    cleaned = clean_boilerplate(text)

    if not check_length(cleaned, min_chars=min_chars, min_words=min_words):
        return None

    if not check_alpha_density(cleaned, min_ratio=min_alpha_ratio):
        return None

    if not check_line_repetition(cleaned, max_ratio=max_line_dup_ratio):
        return None

    if not check_ngram_repetition(cleaned, n=ngram_size, max_ratio=max_ngram_dup_ratio):
        return None

    if is_pt and not check_pt_stopwords(cleaned, min_count=min_pt_stopwords):
        return None

    return cleaned
