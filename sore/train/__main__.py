"""CLI de treino: `python -m sore.train --config X.yaml [--resume ckpt.pt] [--max-steps N]`."""
from __future__ import annotations

import argparse
from pathlib import Path

from sore.train.config import load_train_config
from sore.train.trainer import train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m sore.train", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="YAML do TrainConfig")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint (last.pt) para retomar")
    parser.add_argument("--max-steps", type=int, default=None, help="máximo de passos nesta execução")
    args = parser.parse_args(argv)
    if args.max_steps is not None and args.max_steps < 0:
        parser.error("--max-steps deve ser >= 0")
    train(load_train_config(args.config), resume=args.resume, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
