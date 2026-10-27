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


def _make_procedural_pt_web_doc(i: int) -> dict[str, Any]:
    topics = [
        "redes neurais profundas e mecanismos de atenção convolucional",
        "computação quântica e arquiteturas de supercomputadores modernos",
        "biologia molecular e sequenciamento genético automatizado",
        "energia solar fotovoltaica e transição energética sustentável",
        "agricultura de precisão e sensoriamento remoto por satélites",
        "arqueologia subaquática e preservação do patrimônio histórico",
        "astronomia observacional e telescópios espaciais infravermelhos",
        "economia circular e reciclagem de materiais compósitos industriais",
    ]
    actions = [
        "desenvolvem novas abordagens metodológicas para otimizar",
        "analisam os impactos socioeconômicos decorrentes de",
        "apresentam resultados empíricos consistentes demonstrando",
        "investigam correlações estatísticas significativas entre",
        "propõem modelos preditivos avançados capazes de antecipar",
        "estabelecem diretrizes normativas essenciais para regular",
        "conduzem experimentos laboratoriais controlados avaliando",
        "sintetizam evidências documentais recentes a respeito de",
    ]
    t1 = topics[i % len(topics)]
    t2 = topics[(i + 3) % len(topics)]
    a1 = actions[(i // len(topics)) % len(actions)]
    a2 = actions[(i + 5) % len(actions)]

    text = (
        f"No âmbito dos estudos sobre {t1}, pesquisadores brasileiros {a1} o progresso técnico "
        f"no setor científico número {i}. As análises qualitativas confirmam que {t2} desempenham papel "
        f"relevante na infraestrutura nacional. Ademais, especialistas renomados {a2} as novas descobertas "
        f"com rigor metodológico exemplar. Este documento de registro número {i} sintetiza contribuições "
        f"relevantes para a ciência e tecnologia do país com qualidade e inovação contínua."
    )
    return {"text": text, "is_pt": True}


def _make_procedural_pt_hq_doc(i: int) -> dict[str, Any]:
    disciplines = [
        "A linguística histórica", "A termodinâmica estatística", "A geometria diferencial",
        "A sociologia urbana", "A oceanografia física", "A microeletrônica aplicada",
        "A farmacologia molecular", "A climatologia regional"
    ]
    concepts = [
        "a evolução estrutural dos dialetos ibéricos contemporâneos",
        "o transporte de calor em fluidos não-lineares turbulentos",
        "a curvatura de variedades riemannianas de dimensão arbitrária",
        "a segregação espacial e os padrões de mobilidade coletiva",
        "a circulação termohalina e a dinâmica de massas d'água profundas",
        "a litografia ultravioleta extrema para transistores nanométricos",
        "a interação ligante-receptor em biomembranas lipídicas complexas",
        "as anomalias pluviométricas e teleconexões atmosféricas tropicais"
    ]
    d = disciplines[i % len(disciplines)]
    c = concepts[i % len(concepts)]
    text = (
        f"{d} constitui um ramo do conhecimento voltado a compreender {c} no documento enciclopédico {i}. "
        f"Investigações documentadas atestam que os princípios fundamentais formulados no século passado "
        f"fornecem instrumentos teóricos consistentes para modelar fenômenos complexos na edição {i}. "
        f"Dessa maneira, a sistematização rigorosa de hipóteses e a validação empírica de teses acadêmicas "
        f"asseguram a consolidação do saber científico de alta qualidade em benefício de toda a sociedade."
    )
    return {"text": text, "is_pt": True}


def _make_procedural_en_edu_doc(i: int) -> dict[str, Any]:
    fields = [
        "Numerical linear algebra", "Statistical physics", "Information theory",
        "Abstract algebra", "Differential equations", "Quantum electrodynamics",
        "Graph theory", "Optimization theory"
    ]
    topics = [
        "eigenvalue computation via Lanczos iteration techniques",
        "phase transitions in two-dimensional Ising lattice models",
        "channel capacity bounds under additive white Gaussian noise",
        "homomorphism theorems for finitely generated Abelian groups",
        "existence and uniqueness criteria for boundary value problems",
        "renormalization group flow in perturbative quantum gauge theories",
        "spectral clustering bounds derived from graph Laplacian matrices",
        "primal-dual interior point algorithms for semidefinite programs"
    ]
    f = fields[i % len(fields)]
    t = topics[i % len(topics)]
    text = (
        f"{f} provides an essential mathematical framework for analyzing {t} in instructional unit {i}. "
        f"Theoretical derivations demonstrate that rigorous error bounds and convergence rates can be established "
        f"under standard regularity assumptions throughout lesson module {i}. Consequently, students and "
        f"researchers gain deep understanding of underlying algorithmic principles and computational complexity "
        f"necessary for developing scalable software solutions across diverse scientific domains."
    )
    return {"text": text, "is_pt": False}


def _make_procedural_code_doc(i: int) -> dict[str, Any]:
    funcs = ["transform_vector", "compute_hash_index", "validate_packet", "aggregate_metrics",
             "serialize_state", "interpolate_spline", "decompose_tensor", "filter_frequencies"]
    fn = funcs[i % len(funcs)]
    text = (
        f"def {fn}_v{i}(data_series: list[float], threshold_factor: float = 0.75, offset_val: int = {i}) -> list[float]:\n"
        f"    \"\"\"Performs deterministic numerical transformation for data stream unit {i}.\"\"\"\n"
        f"    processed_buffer: list[float] = []\n"
        f"    for index_counter, sample_value in enumerate(data_series):\n"
        f"        normalized = (sample_value * threshold_factor) + (offset_val % 17)\n"
        f"        if normalized > 0.0 and (index_counter + {i}) % 2 == 0:\n"
        f"            processed_buffer.append(round(normalized, 6))\n"
        f"        else:\n"
        f"            processed_buffer.append(0.0)\n"
        f"    return processed_buffer\n"
    )
    # Utiliza coluna "content" (padrão The Stack v2) para validar fallback de campos
    return {"content": text, "is_pt": False}


def _build_procedural_generator(fn: Any) -> Iterator[dict[str, Any]]:
    """Gera stream contínuo procedural sem colisões de MinHash LSH."""
    idx = 0
    while True:
        yield fn(idx)
        idx += 1


def get_mock_sources() -> dict[str, Iterator[dict[str, Any]]]:
    """Cria fontes sintéticas variadas em PT, EN e Código com geração procedural infinita."""
    return {
        "pt_web": _build_procedural_generator(_make_procedural_pt_web_doc),
        "pt_hq": _build_procedural_generator(_make_procedural_pt_hq_doc),
        "en_edu": _build_procedural_generator(_make_procedural_en_edu_doc),
        "code": _build_procedural_generator(_make_procedural_code_doc),
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
    print(f"Documentos aceitos     : {stats['documents_accepted']}")
    print(f"Documentos duplicados  : {stats['documents_duplicated']}")
    print(f"Documentos filtrados   : {stats['documents_filtered']}")
    rejections = stats.get("filter_rejections", {})
    print("  - Comprimento (length)        :", rejections.get("length", 0))
    print("  - Densidade alfanumérica      :", rejections.get("alpha_density", 0))
    print("  - Repetição de linhas         :", rejections.get("line_repetition", 0))
    print("  - Repetição de n-gramas       :", rejections.get("ngram_repetition", 0))
    print("  - Stopwords em português      :", rejections.get("pt_stopwords", 0))
    print(f"Tokens gravados        : {stats['tokens_written']}")
    print("==================================================")
    for s in shards:
        print(f" - {s.name} ({s.with_suffix('.json').name})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
