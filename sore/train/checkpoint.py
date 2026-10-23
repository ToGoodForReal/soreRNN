"""Checkpoint retomável: modelo, otimizadores, config do modelo, cursor de dados, estado do treino e RNG.

Tudo é salvo em tipos aceitos por `torch.load(weights_only=True)` (tensores, dicts, listas, tuplas,
escalares); o estado do numpy tem a chave convertida de ndarray para lista.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

from sore.model import SoreLM

# Campos de execução do ModelConfig: podem mudar entre o save e a retomada sem mudar os pesos.
# `mtp_weight` entra aqui porque o trainer usa o λ do TrainConfig (R23), não o do modelo.
_EXECUTION_FIELDS = ("gdn_backend", "attn_backend", "grad_checkpoint", "mtp_weight", "loss_chunk_size")


def _rng_state() -> dict:
    np_state = np.random.get_state(legacy=False)
    np_state["state"]["key"] = np_state["state"]["key"].tolist()
    return {
        "python": random.getstate(),
        "numpy": np_state,
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np_state = state["numpy"]
    np_state["state"]["key"] = np.asarray(np_state["state"]["key"], dtype=np.uint32)
    np.random.set_state(np_state)
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    model: SoreLM,
    optimizers: list[torch.optim.Optimizer],
    data_state: dict,
    train_state: dict,
) -> None:
    """Grava em `path.with_suffix(".tmp")` (com fsync) e troca por `path` com `os.replace` (atômico).

    `model` é o módulo sem wrapper (`torch.compile` mudaria os nomes do state_dict).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizers": [opt.state_dict() for opt in optimizers],
        "model_cfg": model.cfg.to_dict(),
        "data_state": data_state,
        "train_state": train_state,
        "rng": _rng_state(),
    }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(path: Path, model: SoreLM, optimizers: list[torch.optim.Optimizer]) -> tuple[dict, dict]:
    """Restaura modelo, otimizadores e RNG in-place; devolve `(data_state, train_state)`.

    ValueError se a arquitetura do checkpoint diferir da do `model` (campos de execução à parte)
    ou se o número de otimizadores não bater.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    saved, current = ckpt["model_cfg"], model.cfg.to_dict()
    diff = sorted(k for k in current if k not in _EXECUTION_FIELDS and saved.get(k) != current[k])
    if diff:
        raise ValueError(f"{path}: arquitetura do checkpoint difere do modelo nos campos {diff}")
    if len(ckpt["optimizers"]) != len(optimizers):
        raise ValueError(f"{path}: {len(ckpt['optimizers'])} otimizadores no checkpoint, {len(optimizers)} no treino")
    model.load_state_dict(ckpt["model"])
    for opt, state in zip(optimizers, ckpt["optimizers"]):
        opt.load_state_dict(state)
    _set_rng_state(ckpt["rng"])
    return ckpt["data_state"], ckpt["train_state"]
