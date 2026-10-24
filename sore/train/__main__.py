"""CLI de treino: `python -m sore.train --config X.yaml [--set KEY=VALUE ...] [--resume ckpt.pt] [--max-steps N]`."""
from __future__ import annotations

import argparse
from pathlib import Path

from sore.train.config import load_train_config, parse_overrides
from sore.train.trainer import train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m sore.train", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="YAML do TrainConfig")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="sobrescreve uma chave do YAML (repetível); `model_overrides.X=...` faz merge com o do YAML",
    )
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint (last.pt) para retomar")
    parser.add_argument("--max-steps", type=int, default=None, help="máximo de passos nesta execução")
    args = parser.parse_args(argv)
    if args.max_steps is not None and args.max_steps < 0:
        parser.error("--max-steps deve ser >= 0")
    try:
        overrides = parse_overrides(args.overrides)
    except ValueError as e:
        parser.error(str(e))
    train(load_train_config(args.config, overrides), resume=args.resume, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
