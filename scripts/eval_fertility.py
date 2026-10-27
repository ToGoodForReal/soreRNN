"""Avaliação da fertilidade (tokens/palavra) de tokenizadores em português.

Mede a fertilidade como a razão entre o total de tokens produzidos pela tokenização
e o total de palavras da amostra textual:
    Fertilidade = total_tokens / total_palavras

Compara o tokenizador podado (96K) com a base de referência (Qwen/Qwen3.5-0.8B com 248K)
e assegura que a degradação na fertilidade em português não exceda 3%:
    (Fertilidade_96k / Fertilidade_248k) <= 1.03
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# Amostra canônica e rica de sentenças em português cobrindo múltiplos domínios
# (noticioso, literário, científico, técnico, formal/jurídico e conversacional).
DEFAULT_PT_SAMPLES: list[str] = [
    "A inteligência artificial tem avançado significativamente nas últimas décadas no mundo inteiro.",
    "Pesquisadores desenvolvem novos modelos de linguagem com arquiteturas eficientes e compactas.",
    "O objetivo é permitir que sistemas inteligentes sejam executados em dispositivos com recursos computacionais finitos.",
    "A engenharia de software contemporânea exige soluções práticas, determinísticas e altamente escaláveis.",
    "O processamento de linguagem natural envolve técnicas estatísticas e redes neurais profundas.",
    "A Constituição da República Federativa do Brasil assegura a dignidade da pessoa humana como fundamento basilar.",
    "Todos são iguais perante a lei, sem distinção de qualquer natureza, garantindo-se a inviolabilidade do direito à vida.",
    "Dom Casmurro é uma das obras mais célebres do escritor brasileiro Machado de Assis.",
    "Capitu olhou para o mar com aqueles olhos de ressaca que pareciam tragar o próprio horizonte infinito.",
    "O algoritmo de ordenação rápida possui complexidade assintótica média de O(n log n).",
    "Estruturas de dados auto-balanceadas mantêm a altura da árvore proporcional ao logaritmo do número de nós.",
    "A fotossíntese é o processo biológico através do qual plantas convertem energia solar em energia química.",
    "As mitocôndrias atuam na respiração celular e produzem trifosfato de adenosina para o metabolismo orgânico.",
    "Em caso de rescisão contratual sem justa causa, haverá incidência de multa rescisória proporcional ao saldo acumulado.",
    "As partes contratantes elegem o foro da Comarca de São Paulo para dirimir quaisquer dúvidas oriundas deste termo.",
    "Bom dia! Gostaria de saber como faço para alterar o endereço cadastrado na minha conta bancária.",
    "Combinamos de almoçar naquele restaurante novo no centro da cidade por volta do meio-dia amanhã.",
    "A inflação acumulada nos últimos doze meses desacelerou conforme indicado pelo índice de preços ao consumidor.",
    "O Banco Central anunciou uma redução gradual na taxa básica de juros para estimular a atividade produtiva.",
    "O compilador otimiza o grafo de computação eliminando subexpressões comuns e fundindo operações contíguas.",
    "A GPU executa milhares de threads simultâneas em unidades de streaming multiprocessor com alta largura de banda.",
    "Os oceanos cobrem mais de setenta por cento da superfície terrestre e regulam o clima global.",
    "A conservação da biodiversidade nas florestas tropicais é imperativa para a estabilidade do ecossistema.",
    "Não hesite em entrar em contato caso surja alguma dúvida adicional sobre as regras deste projeto.",
    "O aprendizado supervisionado depende de grandes bases de dados devidamente anotadas e higienizadas.",
    "Mecanismos de atenção linear combinam eficiência temporal com precisão em modelagem de sequências longas.",
    "A transição energética para matrizes limpas exige investimento maciço em geração eólica e fotovoltaica.",
    "O comitê gestor aprovou as diretrizes orçamentárias preliminares para o próximo exercício fiscal.",
    "Pode deixar que eu mesmo finalizo o relatório técnico e envio para a equipe antes do final da tarde.",
    "O rápido crescimento demográfico nos centros urbanos demanda soluções inteligentes de mobilidade urbana sustentável.",
]


def load_tokenizer(target: Any) -> Any:
    """Carrega um tokenizador a partir de arquivo, diretório, repo HF ou objeto existente.

    Suporta:
    - Objeto que já implementa o método `.encode` (MockTokenizer, Tokenizer, etc.)
    - Caminho direto para arquivo `tokenizer.json`
    - Diretório contendo `tokenizer.json`
    - Identificador de repositório Hugging Face (e.g. `Qwen/Qwen3.5-0.8B`)
    """
    if isinstance(target, (str, Path)):
        target_str = str(target)
        target_path = Path(target_str)

        # 1. Caminho direto para arquivo
        if target_path.is_file():
            from tokenizers import Tokenizer
            return Tokenizer.from_file(str(target_path))

        # 2. Caminho de diretório com tokenizer.json
        if target_path.is_dir() and (target_path / "tokenizer.json").is_file():
            from tokenizers import Tokenizer
            return Tokenizer.from_file(str(target_path / "tokenizer.json"))

        # 3. Identificador de modelo Hugging Face via tokenizers
        try:
            from tokenizers import Tokenizer
            return Tokenizer.from_pretrained(target_str)
        except Exception:
            pass

        # 4. Fallback via transformers AutoTokenizer
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(target_str)
        except Exception as exc:
            raise ValueError(
                f"Não foi possível carregar o tokenizador de '{target_str}': {exc}"
            ) from exc

    if hasattr(target, "encode"):
        return target

    raise TypeError(f"Tipo de tokenizador inválido ou incompatível: {type(target)}")



def count_tokens(tokenizer: Any, text: str) -> int:
    """Codifica o texto com o tokenizador e devolve a contagem exata de tokens."""
    encoded = tokenizer.encode(text)
    if hasattr(encoded, "ids"):
        return len(encoded.ids)
    if isinstance(encoded, list):
        return len(encoded)
    return len(encoded)


def count_words(text: str) -> int:
    """Conta a quantidade de palavras no texto divididas por espaços em branco."""
    return len(text.split())


def calculate_fertility(tokenizer: Any, texts: list[str]) -> tuple[float, int, int]:
    """Calcula a fertilidade agregada (tokens/palavras) em uma coleção de textos.

    Retorna:
        (fertilidade, total_tokens, total_palavras)
    """
    total_tokens = 0
    total_words = 0

    for t in texts:
        t_clean = t.strip()
        if not t_clean:
            continue
        n_words = count_words(t_clean)
        if n_words == 0:
            continue
        n_tokens = count_tokens(tokenizer, t_clean)
        total_tokens += n_tokens
        total_words += n_words

    if total_words == 0:
        raise ValueError("A amostra textual não contém nenhuma palavra válida.")

    fertility = total_tokens / total_words
    return fertility, total_tokens, total_words


def evaluate_fertility(
    base_tokenizer: Any,
    pruned_tokenizer: Any,
    texts: list[str],
    max_degradation: float = 0.03,
) -> dict[str, Any]:
    """Compara a fertilidade do tokenizador podado contra a base de referência.

    Parâmetros:
        base_tokenizer: Tokenizador base (248K ou referência).
        pruned_tokenizer: Tokenizador podado (96K).
        texts: Lista de sentenças ou parágrafos em português para avaliação.
        max_degradation: Degradação máxima permitida (padrão: 0.03 = 3%).

    Retorna:
        Dicionário com métricas calculadas e indicador de aprovação no portão de qualidade.
    """
    base_tok = load_tokenizer(base_tokenizer)
    pruned_tok = load_tokenizer(pruned_tokenizer)

    base_fert, base_tokens, total_words = calculate_fertility(base_tok, texts)
    pruned_fert, pruned_tokens, _ = calculate_fertility(pruned_tok, texts)

    ratio = pruned_fert / base_fert
    degradation = ratio - 1.0
    passed = ratio <= (1.0 + max_degradation + 1e-9)

    return {
        "base_fertility": base_fert,
        "pruned_fertility": pruned_fert,
        "ratio": ratio,
        "degradation_pct": degradation * 100.0,
        "max_allowed_degradation_pct": max_degradation * 100.0,
        "passed": passed,
        "total_words": total_words,
        "base_tokens": base_tokens,
        "pruned_tokens": pruned_tokens,
        "num_sentences": len([t for t in texts if t.strip()]),
    }


def load_texts(
    text_file: str | Path | None = None,
    sample_pt: str | Path | None = None,
    max_sentences: int | None = None,
) -> list[str]:
    """Obtém as sentenças a serem avaliadas a partir de arquivo ou da amostra interna."""
    texts: list[str] = []

    if text_file is not None:
        p = Path(text_file)
        if not p.is_file():
            raise FileNotFoundError(f"Arquivo de texto não encontrado: {text_file}")
        texts = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif sample_pt is not None and sample_pt != "default":
        p = Path(sample_pt)
        if p.is_file():
            texts = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            texts = list(DEFAULT_PT_SAMPLES)
    else:
        texts = list(DEFAULT_PT_SAMPLES)

    if max_sentences is not None and max_sentences > 0:
        texts = texts[:max_sentences]

    return texts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia a fertilidade (tokens/palavra) do tokenizador podado em comparação à base."
    )
    parser.add_argument(
        "--base-tokenizer",
        type=str,
        default="Qwen/Qwen3.5-0.8B",
        help="Caminho local ou repo HuggingFace do tokenizador base de referência.",
    )
    parser.add_argument(
        "--pruned-tokenizer",
        type=str,
        default=None,
        help="Caminho local ou repo HuggingFace do tokenizador podado (96K).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Executa teste rápido com tokenizadores sintéticos locais sem baixar modelos.",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="Arquivo de texto com sentenças em português para avaliação.",
    )
    parser.add_argument(
        "--sample-pt",
        nargs="?",
        const="default",
        default=None,
        help="Usa amostra de sentenças em português (caminho opcional ou amostra padrão embutida).",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Quantidade máxima de sentenças a serem avaliadas.",
    )
    parser.add_argument(
        "--max-degradation",
        type=float,
        default=0.03,
        help="Degradação máxima aceitável de fertilidade (padrão: 0.03 = 3%%).",
    )
    parser.add_argument(
        "--assert-ratio",
        action="store_true",
        help="Levanta AssertionError se a degradação ultrapassar o limite aceitável de 3%%.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    texts = load_texts(
        text_file=args.text_file,
        sample_pt=args.sample_pt,
        max_sentences=args.max_sentences,
    )

    if args.mock:
        class MockEvalTokenizer:
            def __init__(self, mult: float):
                self.mult = mult

            def encode(self, text: str) -> list[int]:
                n = len(text.split())
                return list(range(max(1, round(n * self.mult))))

        base_tokenizer = MockEvalTokenizer(1.0)
        pruned_tokenizer = MockEvalTokenizer(1.01)
        base_desc = "MockBaseTokenizer (1.0x)"
        pruned_desc = "MockPrunedTokenizer (1.01x)"
    else:
        if args.pruned_tokenizer is None:
            raise ValueError("O argumento --pruned-tokenizer é obrigatório quando não estiver usando --mock.")
        base_tokenizer = args.base_tokenizer
        pruned_tokenizer = args.pruned_tokenizer
        base_desc = str(base_tokenizer)
        pruned_desc = str(pruned_tokenizer)

    print(f"Iniciando avaliação de fertilidade com {len(texts)} sentenças em português...")
    print(f"Tokenizador Base:   {base_desc}")
    print(f"Tokenizador Podado: {pruned_desc}")

    result = evaluate_fertility(
        base_tokenizer=base_tokenizer,
        pruned_tokenizer=pruned_tokenizer,
        texts=texts,
        max_degradation=args.max_degradation,
    )

    status_str = "APROVADO (<= 3%)" if result["passed"] else "REPROVADO (> 3%)"
    print("\n" + "=" * 65)
    print("RELATÓRIO DE FERTILIDADE EM PORTUGUÊS")
    print("=" * 65)
    print(f"Sentenças avaliadas:      {result['num_sentences']}")
    print(f"Total de palavras:        {result['total_words']}")
    print(f"Tokens Base (248K):       {result['base_tokens']}")
    print(f"Tokens Podado (96K):      {result['pruned_tokens']}")
    print(f"Fertilidade Base:         {result['base_fertility']:.4f} tokens/palavra")
    print(f"Fertilidade Podado:       {result['pruned_fertility']:.4f} tokens/palavra")
    print(f"Razão (Podado / Base):    {result['ratio']:.4f}")
    print(f"Degradação Percentual:    {result['degradation_pct']:+.2f}%")
    print(f"Limite Máximo Permitido:  +{result['max_allowed_degradation_pct']:.2f}%")
    print(f"Status do Portão:         {status_str}")
    print("=" * 65)

    if not result["passed"]:
        msg = (
            f"Portão de Fertilidade violado: degradação {result['degradation_pct']:.2f}% "
            f"excede o limite de {result['max_allowed_degradation_pct']:.2f}%."
        )
        if args.assert_ratio:
            raise AssertionError(msg)
        print(f"ERRO: {msg}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
