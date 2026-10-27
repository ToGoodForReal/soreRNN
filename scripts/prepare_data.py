"""CLI para preparação de dados de pré-treino em streaming e particionamento de shards.

Consome fluxos balanceados (PT Web, PT HQ, EN Edu, Código), aplica filtros heurísticos,
deduplicação MinHash LSH, tokenização e gravação contígua em shards uint32.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Iterator

from sore.data.pipeline import run_pipeline


class SimpleMockTokenizer:
    """Tokenizador sintético determinístico para testes e validações offline."""

    def __init__(self, vocab_size: int = 98304) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        words = text.split()
        return [abs(hash(w)) % (self.vocab_size - 1) for w in words]


def _build_mock_generator(docs: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Gera stream contínuo ciclando sobre a lista de documentos fornecida."""
    while True:
        for d in docs:
            yield dict(d)


def get_mock_sources() -> dict[str, Iterator[dict[str, Any]]]:
    """Cria fontes sintéticas variadas em PT, EN e Código que satisfazem os filtros de qualidade."""
    pt_web_base = [
        {
            "text": (
                "A inteligência artificial generativa e os modelos de linguagem em grande escala "
                "transformaram profundamente o cenário científico e tecnológico contemporâneo em todo o país. "
                "Pesquisadores brasileiros dedicam-se ao desenvolvimento de arquiteturas mais eficientes, compactas "
                "e especializadas no português brasileiro, promovendo soberania tecnológica e inclusão digital ampla."
            ),
            "is_pt": True,
        },
        {
            "text": (
                "O treinamento de redes neurais profundas requer infraestrutura de alto desempenho e grandes "
                "volumes de dados textuais cuidadosamente tratados e limpos. A correta higienização do corpus, "
                "incluindo a eliminação de ruídos e repetições indesejadas, é fundamental para garantir a convergência "
                "estável do modelo e a coerência das respostas geradas pelo assistente."
            ),
            "is_pt": True,
        },
    ]

    pt_hq_base = [
        {
            "text": (
                "A ciência da computação teórica estuda a complexidade de algoritmos e a capacidade intrínseca "
                "dos autômatos formais e sistemas de transição de estados. Na linguística computacional, "
                "representações probabilísticas e modelos de n-gramas formaram a base para o desenvolvimento "
                "dos atuais transformadores e modelos neurais autorregressivos de alta fidelidade semântica."
            ),
            "is_pt": True,
        },
        {
            "text": (
                "O bioma da Amazônia abriga a maior biodiversidade tropical do planeta Terra, com milhares "
                "de espécies endêmicas de flora e fauna catalogadas por instituições científicas nacionais. "
                "A conservação ambiental e o manejo florestal sustentável são temas de relevância global para "
                "a mitigação das mudanças climáticas e o equilíbrio hídrico continental da América do Sul."
            ),
            "is_pt": True,
        },
    ]

    en_edu_base = [
        {
            "text": (
                "Linear dynamical systems and state space models define a powerful mathematical foundation "
                "for sequential data processing in continuous time. By discretizing state transition equations "
                "with bilinear transforms, neural architectures can achieve constant recurrent state complexity "
                "during token generation and linear training cost with respect to context window length."
            ),
            "is_pt": False,
        },
        {
            "text": (
                "Matrix decomposition methods such as singular value decomposition and eigenvalue factorization "
                "provide essential analytical tools for dimensionality reduction in machine learning pipelines. "
                "These transformations preserve dominant variance directions while eliminating redundant numerical "
                "projections across high-dimensional feature spaces."
            ),
            "is_pt": False,
        },
    ]

    code_base = [
        {
            "text": (
                "def binary_search_insert_position(sorted_array: list[int], target: int) -> int:\n"
                "    \"\"\"Finds the insertion index for target in a sorted list using binary search.\"\"\"\n"
                "    low, high = 0, len(sorted_array)\n"
                "    while low < high:\n"
                "        mid = (low + high) // 2\n"
                "        if sorted_array[mid] < target:\n"
                "            low = mid + 1\n"
                "        else:\n"
                "            high = mid\n"
                "    return low\n"
            ),
            "is_pt": False,
        },
        {
            "text": (
                "class AccumulatorBuffer:\n"
                "    \"\"\"Thread-safe rolling accumulator for continuous streaming metrics.\"\"\"\n"
                "    def __init__(self, max_size: int = 1000):\n"
                "        self.max_size = max_size\n"
                "        self.values: list[float] = []\n"
                "    def push(self, val: float) -> None:\n"
                "        if len(self.values) >= self.max_size:\n"
                "            self.values.pop(0)\n"
                "        self.values.append(val)\n"
            ),
            "is_pt": False,
        },
    ]

    return {
        "pt_web": _build_mock_generator(pt_web_base),
        "pt_hq": _build_mock_generator(pt_hq_base),
        "en_edu": _build_mock_generator(en_edu_base),
        "code": _build_mock_generator(code_base),
    }


def load_tokenizer(tokenizer_path: str, is_mock: bool = False) -> Any:
    """Carrega o tokenizador do disco ou utiliza o mock sintético para testes offline."""
    p = Path(tokenizer_path)
    json_path = p if p.is_file() else p / "tokenizer.json"
    if json_path.exists():
        from tokenizers import Tokenizer
        return Tokenizer.from_file(str(json_path))
    if is_mock:
        return SimpleMockTokenizer(vocab_size=98304)
    raise FileNotFoundError(
        f"Tokenizer não encontrado em {json_path}. Execute scripts/prune_tokenizer.py primeiro ou utilize a flag --mock."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Configura e processa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="CLI para ingestão em streaming, filtros, deduplicação e sharding de dados de pré-treino."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/shards",
        help="Diretório onde os shards binários .bin e sidecars .json serão gravados.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=268435456,
        help="Quantidade máxima de tokens a serem gravados no total.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=268435456,
        help="Capacidade máxima de cada shard em tokens uint32 (padrão: 2^28 = 268.435.456, ~1 GiB).",
    )
    parser.add_argument(
        "--unit-len",
        type=int,
        default=None,
        help="Tamanho do bloco unit_len de empacotamento em tokens (padrão: 8192 ou divisível por shard_size).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="artifacts/tokenizer_96k",
        help="Caminho para o diretório ou arquivo do tokenizador podado.",
    )
    parser.add_argument(
        "--eos-id",
        type=int,
        default=98303,
        help="ID do token especial EOS gravado ao final de cada documento.",
    )
    parser.add_argument(
        "--lsh-capacity",
        type=int,
        default=100000,
        help="Capacidade do buffer circular do MinHash LSH para deduplicação.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Usa fontes sintéticas e tokenizador mock para testes offline sem rede.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente determinística pseudo-aleatória.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada principal da CLI de preparação de dados."""
    args = parse_args(argv)
    tokenizer = load_tokenizer(args.tokenizer_path, is_mock=args.mock)

    weights = {
        "pt_web": 0.65,
        "pt_hq": 0.10,
        "en_edu": 0.15,
        "code": 0.10,
    }

    if args.mock:
        sources = get_mock_sources()
    else:
        try:
            from datasets import load_dataset
        except ImportError as err:
            raise ImportError(
                "A biblioteca 'datasets' da Hugging Face é necessária para ingestão de corpus real. "
                "Instale com 'uv add datasets' ou use o parâmetro --mock."
            ) from err

        sources = {
            "pt_web": load_dataset("HuggingFaceFW/fineweb-2", name="por_Latn", split="train", streaming=True),
            "pt_hq": load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True),
            "en_edu": load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True),
            "code": load_dataset("bigcode/the-stack-v2-train-smol", split="train", streaming=True),
        }

    print("Iniciando pipeline de processamento e sharding de dados:")
    print(f"  Diretório de saída: {args.output_dir}")
    print(f"  Max Tokens        : {args.max_tokens}")
    print(f"  Shard Size        : {args.shard_size}")
    print(f"  Modo Mock         : {args.mock}")

    result = run_pipeline(
        sources=sources,
        weights=weights,
        tokenizer=tokenizer,
        out_dir=args.output_dir,
        max_tokens=args.max_tokens,
        shard_size=args.shard_size,
        unit_len=args.unit_len,
        eos_id=args.eos_id,
        lsh_capacity=args.lsh_capacity,
        seed=args.seed,
        show_progress=True,
        return_stats=True,
    )
    shards, stats = result  # type: ignore[misc]

    print("\n================ Pipeline Summary ================")
    print(f"Diretório de saída     : {args.output_dir}")
    print(f"Shards gerados         : {len(shards)}")
    print(f"Documentos lidos       : {stats['documents_seen']}")
    print(f"Documentos filtrados   : {stats['documents_filtered']}")
    print(f"Documentos duplicados  : {stats['documents_duplicated']}")
    print(f"Documentos aceitos     : {stats['documents_accepted']}")
    print(f"Tokens gravados        : {stats['tokens_written']}")
    print("==================================================")
    for s in shards:
        print(f" - {s.name} ({s.with_suffix('.json').name})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
