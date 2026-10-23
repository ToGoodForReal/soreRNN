"""Loop de pré-treino retomável.

Cada passo busca os `grad_accum` micro-batches antes do forward, conta os tokens válidos na CPU e
faz o backward de `loss_mb * n_mb / n_passo`: o loss do passo é a média ponderada por token. Depois
vêm o clip do grad norm, a guarda de passo não finito (zera os grads e não anda nenhum otimizador),
o lr `initial_lr * wsd(step)` e o passo dos otimizadores. O estágio do currículo de contexto é o
primeiro com `until_tokens > tokens_consumed`, avaliado no início de cada passo.
"""
from __future__ import annotations

import contextlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sore.model import SoreLM, get_preset
from sore.model.loss import IGNORE_INDEX
from sore.train.checkpoint import load_checkpoint, save_checkpoint
from sore.train.config import Stage, TrainConfig, resolve_data_paths
from sore.train.data import PackedTokenStream
from sore.train.optim import build_optimizers
from sore.train.schedule import wsd_multiplier


@dataclass
class TrainSummary:
    steps: int
    tokens: int
    skipped_steps: int
    last_loss: float


def resolve_device(device: str) -> torch.device:
    """`"auto"` vira `"cuda"` se houver GPU, senão `"cpu"`."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def active_stage(stages: list[Stage], tokens: int) -> Stage:
    """Primeiro estágio com `until_tokens > tokens` (o último, se todos já passaram)."""
    return next((s for s in stages if s.until_tokens > tokens), stages[-1])


def _sync_stage(stream: PackedTokenStream, stages: list[Stage]) -> bool:
    """Ajusta o formato do stream ao estágio ativo; True se mudou.

    `set_shape` volta a posição a 0, então só é chamado quando o formato muda de fato: depois de um
    `load_state_dict` no meio de um estágio, o stream continua de onde parou.
    """
    stage = active_stage(stages, stream.tokens_consumed)
    if (stream.seq_len, stream.micro_batch) == (stage.seq_len, stage.micro_batch):
        return False
    stream.set_shape(stage.seq_len, stage.micro_batch)
    return True


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()  # CPU: fp32


def _train_step(fwd, model, optimizers, stream, cfg: TrainConfig, device, lr_mult: float) -> dict:
    """Um passo de otimização; devolve as métricas do passo (médias ponderadas por token)."""
    batches = [next(stream) for _ in range(cfg.grad_accum)]
    n_valid = [int((b.labels != IGNORE_INDEX).sum()) for b in batches]
    n_step = sum(n_valid)
    # loss, lm_loss, mtp_loss; passo sem nenhum token válido não tem loss (NaN -> null no log)
    sums = torch.zeros(3, device=device) if n_step else torch.full((3,), math.nan, device=device)
    for batch, n in zip(batches, n_valid):
        if n == 0:
            continue
        weight = n / n_step
        with _autocast(device):
            out = fwd(
                batch.input_ids.to(device, non_blocking=True),
                cu_seqlens=batch.cu_seqlens.to(device, non_blocking=True),
                labels=batch.labels.to(device, non_blocking=True),
            )
        (out.loss * weight).backward()
        mtp = out.mtp_loss if out.mtp_loss is not None else torch.full_like(out.loss, math.nan)  # sem MTP -> null
        sums += weight * torch.stack([t.detach() for t in (out.loss, out.lm_loss, mtp)])

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    loss, lm_loss, mtp_loss, norm = torch.cat([sums, grad_norm.reshape(1).to(sums)]).tolist()  # 1 sync
    finite = math.isfinite(norm)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lr_mult
        if finite:
            opt.step()
        opt.zero_grad(set_to_none=True)
    return {"loss": loss, "lm_loss": lm_loss, "mtp_loss": mtp_loss, "grad_norm": norm, "skipped": not finite}


def _json_float(x: float) -> float | None:
    return x if math.isfinite(x) else None  # JSON estrito: NaN/Inf viram null


def train(cfg: TrainConfig, resume: Path | None = None, max_steps: int | None = None) -> TrainSummary:
    """Treina até `total_steps` ou até `max_steps` passos nesta chamada (o que vier antes).

    `resume` restaura modelo, otimizadores, RNG, cursor de dados e contadores de um `last.pt`.
    """
    device = resolve_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = SoreLM(get_preset(cfg.model_preset, **cfg.model_overrides)).to(device)
    # otimizadores sobre o módulo sem wrapper: a divisão Muon/AdamW depende dos nomes dos parâmetros
    optimizers = build_optimizers(model, cfg.lr_muon, cfg.lr_adamw, cfg.weight_decay)
    fwd = model
    if cfg.compile:
        # torch 2.14.1: a fusão "mix order reduction" do Inductor quebra no 1º backward do SoreLM
        # (InductorError: AssertionError "509 v.s. 510" em MixOrderReduction.get_numel_rnumel).
        # Desligada até o fix upstream; equivale a TORCHINDUCTOR_MIX_ORDER_REDUCTION=0.
        import torch._inductor.config as inductor_config

        inductor_config.triton.mix_order_reduction = False
        fwd = torch.compile(model)
    first = cfg.stages[0]
    stream = PackedTokenStream(
        resolve_data_paths(cfg.data_paths), first.seq_len, first.micro_batch, cfg.eos_id, cfg.seed
    )

    step, skipped, last_loss = 0, 0, math.nan
    if resume is not None:
        data_state, train_state = load_checkpoint(resume, model, optimizers)
        stream.load_state_dict(data_state)
        step, skipped, last_loss = train_state["step"], train_state["skipped_steps"], train_state["last_loss"]

    def checkpoint() -> None:
        train_state = {
            "step": step,
            "skipped_steps": skipped,
            "tokens": stream.tokens_consumed,
            "last_loss": last_loss,
            "config": cfg.to_dict(),
        }
        save_checkpoint(out_dir / "last.pt", model, optimizers, stream.state_dict(), train_state)

    total_steps = cfg.total_steps
    print(
        f"[train] {cfg.model_preset} {model.num_params() / 1e6:.1f}M params | {device} | "
        f"{total_steps} passos x {cfg.tokens_per_step} tokens | início no passo {step}",
        flush=True,
    )
    attempts, saved_at = 0, None
    with open(out_dir / "metrics.jsonl", "a") as log:
        while step < total_steps and (max_steps is None or attempts < max_steps):
            if _sync_stage(stream, cfg.stages):
                print(f"[stage] tokens={stream.tokens_consumed}: seq_len={stream.seq_len} "
                      f"micro_batch={stream.micro_batch}", flush=True)
            t0, tokens_before = time.perf_counter(), stream.tokens_consumed
            lr_mult = wsd_multiplier(step, cfg.warmup_steps, total_steps, cfg.decay_frac)
            stats = _train_step(fwd, model, optimizers, stream, cfg, device, lr_mult)
            step, attempts = step + 1, attempts + 1
            skipped += stats["skipped"]
            last_loss = stats["loss"]
            tok_s = (stream.tokens_consumed - tokens_before) / (time.perf_counter() - t0)
            lr = optimizers[0].param_groups[0]["lr"]
            mem_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0

            record = {
                "step": step,
                "tokens": stream.tokens_consumed,
                **{k: _json_float(stats[k]) for k in ("loss", "lm_loss", "mtp_loss", "grad_norm")},
                "lr_muon": lr,
                "tok_s": tok_s,
                "mem_gb": mem_gb,
                "seq_len": stream.seq_len,
                "skipped": stats["skipped"],
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % cfg.log_every == 0 or stats["skipped"]:
                print(
                    f"step {step}/{total_steps} | loss {stats['loss']:.4f} (lm {stats['lm_loss']:.4f} "
                    f"mtp {stats['mtp_loss']:.4f}) | gnorm {stats['grad_norm']:.3f} | lr {lr:.2e} | "
                    f"{tok_s:,.0f} tok/s | {mem_gb:.2f} GB | seq {stream.seq_len}"
                    + (" | SKIPPED (grad norm não finito)" if stats["skipped"] else ""),
                    flush=True,
                )
            if stream.tokens_consumed // cfg.ckpt_every_tokens > tokens_before // cfg.ckpt_every_tokens:
                checkpoint()
                saved_at = step
    if saved_at != step:
        checkpoint()

    summary = TrainSummary(steps=step, tokens=stream.tokens_consumed, skipped_steps=skipped, last_loss=last_loss)
    print(f"[train] fim: {summary}", flush=True)
    return summary
