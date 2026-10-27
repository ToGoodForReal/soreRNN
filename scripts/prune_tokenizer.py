"""CLI para poda de tokenizador BPE e exportação para HuggingFace com tensores de destilação."""

import argparse

from sore.data.prune_tokenizer import export_pruned_tokenizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poda o vocabulário de um tokenizador BPE mantendo fechamento estrito e exporta para HuggingFace."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Caminho do tokenizador original (arquivo tokenizer.json ou diretório HF).",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Diretório de destino para o tokenizador podado.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=98304,
        help="Tamanho exato do vocabulário podado (padrão 98304).",
    )
    parser.add_argument(
        "--freqs",
        type=str,
        default=None,
        help="Caminho opcional para arquivo de frequências de tokens (JSON, PT ou NPY).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    export_pruned_tokenizer(
        source_tokenizer_path=args.source,
        output_dir=args.out,
        target_size=args.target_size,
        frequencies_path=args.freqs,
    )
    print(f"Tokenizador podado com sucesso ({args.target_size} tokens) salvo em: {args.out}")


if __name__ == "__main__":
    main()
