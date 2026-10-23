"""Divisão de otimizadores: Muon nas matrizes ocultas 2-D, AdamW em todo o resto."""
from __future__ import annotations

import torch
from torch import nn

# Residuais (hyper-connections) ficam no AdamW mesmo quando 2-D: são projeções pequenas e
# especiais (init em zero/uniforme), não matrizes ocultas do tronco.
_RESIDUAL_MARKERS = ("res_mixer", "res_ffn")


def _is_residual(name: str) -> bool:
    return name.startswith("stream_final") or any(m in name for m in _RESIDUAL_MARKERS)


def _goes_to_muon(name: str, p: nn.Parameter) -> bool:
    return (
        p.ndim == 2
        and name.startswith(("blocks.", "mtp."))
        and "conv" not in name
        and not _is_residual(name)
    )


def split_param_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Devolve `(muon, adamw)`: cada parâmetro treinável cai em exatamente um dos dois.

    Muon recebe as matrizes 2-D dentro de `blocks.`/`mtp.` (projeções, FFN), exceto conv e
    residuais. AdamW recebe o resto: embedding, lm_head, normas, `A_log`, `dt_bias`, conv,
    residuais e biases.
    """
    muon: list[nn.Parameter] = []
    adamw: list[nn.Parameter] = []
    for name, p in model.named_parameters():  # named_parameters já deduplica por identidade
        if not p.requires_grad:
            continue
        (muon if _goes_to_muon(name, p) else adamw).append(p)
    return muon, adamw


def build_optimizers(
    model: nn.Module, lr_muon: float, lr_adamw: float, weight_decay: float
) -> list[torch.optim.Optimizer]:
    """Cria `[muon, adamw]` (Muon é omitido se não houver matrizes para ele).

    Cada param group guarda o LR base em `group["initial_lr"]`; o trainer define
    `group["lr"] = group["initial_lr"] * wsd_multiplier(...)` a cada passo.
    """
    muon_params, adamw_params = split_param_groups(model)
    optimizers: list[torch.optim.Optimizer] = []

    if muon_params:
        # match_rms_adamw multiplica o lr por 0.2 * sqrt(max(A, B)) (Moonlight): o update do Muon
        # fica com o RMS de um update do AdamW, então `lr_muon` é um lr na escala do AdamW
        # (default 3e-3, igual ao `lr_adamw`). O 0.02 da literatura vale para adjust_lr_fn="original".
        optimizers.append(
            torch.optim.Muon(
                [{"params": muon_params, "initial_lr": lr_muon}],
                lr=lr_muon,
                weight_decay=weight_decay,
                momentum=0.95,
                nesterov=True,
                ns_steps=5,
                adjust_lr_fn="match_rms_adamw",
            )
        )

    if adamw_params:
        # decay só em ndim >= 2; normas, biases e parâmetros escalares/vetoriais ficam sem decay
        decay = [p for p in adamw_params if p.ndim >= 2]
        no_decay = [p for p in adamw_params if p.ndim < 2]
        groups = [
            {"params": ps, "weight_decay": wd, "initial_lr": lr_adamw}
            for ps, wd in ((decay, weight_decay), (no_decay, 0.0))
            if ps
        ]
        optimizers.append(
            torch.optim.AdamW(
                groups,
                lr=lr_adamw,
                betas=(0.9, 0.95),
                eps=1e-8,
                fused=all(p.is_cuda for p in adamw_params),
            )
        )
    return optimizers
