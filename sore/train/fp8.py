"""Treino FP8 via torchao (alavanca G).

Aplica quantização FP8 em tempo de treino aos módulos `nn.Linear` grandes sob `blocks.` e `mtp.`,
preservando normas, embedding, cabeça de LM e camadas residuais. Em GPUs sem suporte nativo
(capacidade de computação < 8.9), a execução sem emulação levanta RuntimeError.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import torch
import torch.nn as nn
from torchao.float8 import (
    Float8LinearConfig,
    Float8LinearRecipeName,
    convert_to_float8_training,
)
from torchao.float8.float8_linear import Float8Linear


def fp8_supported(device: torch.device | str | None = None) -> bool:
    """Verifica se o dispositivo possui suporte a FP8 nativo (CUDA com capacidade >= 8.9)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if device.type != "cuda" or not torch.cuda.is_available():
        return False

    try:
        cap = torch.cuda.get_device_capability(device)
        return cap >= (8, 9)
    except Exception:
        return False


def _is_fp8_eligible(mod: nn.Module, fqn: str) -> bool:
    """Verifica se o módulo nn.Linear atende ao filtro de conversão FP8.

    Apenas módulos nn.Linear sob `blocks.` e `mtp.` com dimensões múltiplas de 16
    e fora de `res_mixer`/`res_ffn` são elegíveis.
    """
    if not isinstance(mod, nn.Linear):
        return False
    if not (fqn.startswith("blocks.") or fqn.startswith("mtp.")):
        return False
    if "res_mixer" in fqn or "res_ffn" in fqn:
        return False
    return mod.in_features % 16 == 0 and mod.out_features % 16 == 0


def apply_fp8(model: nn.Module, recipe: str, emulate: bool = False) -> int:
    """Converte camadas nn.Linear elegíveis para Float8Linear usando torchao.

    Args:
        model: O modelo SoreLM a converter.
        recipe: Receita de escala ('tensorwise' ou 'rowwise').
        emulate: Se True, simula FP8 mesmo sem aceleração de hardware.

    Returns:
        Número de camadas convertidas para FP8.

    Raises:
        ValueError: Se a receita não for 'tensorwise' ou 'rowwise'.
        RuntimeError: Se emulate=False e o dispositivo não tiver suporte a FP8 (sm89+).
    """
    if recipe not in ("tensorwise", "rowwise"):
        raise ValueError(f"receita FP8 inválida: {recipe!r}; use 'tensorwise' ou 'rowwise'")

    if recipe == "tensorwise":
        import warnings
        warnings.warn(
            "A receita FP8 'tensorwise' sofre de instabilidade numérica sob torch.compile "
            "e grad_checkpoint (produz gradientes NaN no backward no torchao 0.18/cu130 em sm89). "
            "Recomenda-se usar 'rowwise' ou desativar FP8 ('off').",
            UserWarning,
            stacklevel=2,
        )

    param = next(model.parameters(), None)
    device = param.device if param is not None else torch.device("cpu")

    if not emulate and not fp8_supported(device):
        raise RuntimeError(
            f"FP8 exige GPU CUDA com capacidade de computação >= (8, 9), "
            f"mas o dispositivo atual é {device} (capacidade não suportada)"
        )

    recipe_enum = (
        Float8LinearRecipeName.TENSORWISE
        if recipe == "tensorwise"
        else Float8LinearRecipeName.ROWWISE
    )
    base_cfg = Float8LinearConfig.from_recipe_name(recipe_enum)
    f8_cfg = dataclasses.replace(base_cfg, emulate=emulate)

    converted_count = 0

    def module_filter_fn(mod: nn.Module, fqn: str) -> bool:
        nonlocal converted_count
        if _is_fp8_eligible(mod, fqn):
            converted_count += 1
            return True
        return False

    convert_to_float8_training(model, module_filter_fn=module_filter_fn, config=f8_cfg)
    return converted_count
