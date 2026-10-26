"""Loop de pré-treino retomável.

Cada passo busca os `grad_accum` micro-batches antes do forward, conta os tokens válidos na CPU,
copia todos para a GPU (pinned, `non_blocking`) e faz o backward de `loss_mb * n_mb / n_passo`,
com `loss_mb = lm_loss + λ(passo) * mtp_loss` (λ linear de `mtp_weight_start` a `mtp_weight_end`,
R23): o loss do passo é a média ponderada por token. Depois vêm o clip do grad norm, a guarda de
passo não finito (zera os grads e não anda nenhum otimizador), o lr `initial_lr * wsd(step)` e o
passo dos otimizadores. O estágio do currículo de contexto é o primeiro com
`until_tokens > tokens_consumed`, avaliado no início de cada passo. Depois do warmup, um loss
acima de `(1 + spike_threshold)` × a EMA do loss marca `"spike": true` no metrics.jsonl e imprime
um aviso.

Checkpoints (R27): `last.pt` a cada `ckpt_every_tokens`, snapshots numerados a cada
`snapshot_every_tokens` (os `keep_snapshots` mais novos ficam) e `last.pt` na fronteira do passo
depois de um SIGINT/SIGTERM.
"""
from __future__ import annotations

import contextlib
import json
import math
import random
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sore.model import SoreLM, get_preset
from sore.model.flops import flops_per_token
from sore.model.loss import IGNORE_INDEX
from sore.train.checkpoint import load_checkpoint, save_checkpoint
from sore.train.config import Stage, TrainConfig, resolve_data_paths
from sore.train.data import PackedTokenStream
from sore.train.optim import apply_config_hparams, build_optimizers
from sore.train.schedule import linear_schedule, wsd_multiplier


@dataclass
class TrainSummary:
    steps: int
    tokens: int
    skipped_steps: int
    last_loss: float


class LossSpikeDetector:
    """EMA do loss do passo; `detector(loss)` diz se é spike (`loss > ema * (1 + threshold)`).

    A 1a chamada só inicializa a EMA. Spikes e losses não finitos não a atualizam.
    """

    def __init__(self, threshold: float, beta: float = 0.98, ema: float | None = None) -> None:
        self.threshold = threshold
        self.beta = beta
        self.ema = ema

    def __call__(self, loss: float) -> bool:
        if not math.isfinite(loss):
            return False
        if self.ema is not None and loss > self.ema * (1 + self.threshold):
            return True
        self.ema = loss if self.ema is None else self.beta * self.ema + (1 - self.beta) * loss
        return False


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

    `set_shape` avança a posição até a próxima unidade inteira, então só é chamado quando o formato
    muda de fato: depois de um `load_state_dict` no meio de um estágio, o stream continua de onde parou.
    """
    stage = active_stage(stages, stream.tokens_consumed)
    if (stream.seq_len, stream.micro_batch) == (stage.seq_len, stage.micro_batch):
        return False
    stream.set_shape(stage.seq_len, stage.micro_batch)
    return True


def compile_model(model: SoreLM, mode: str | None = None) -> SoreLM:
    """Compila cada Block e a CE da loss in-place (`nn.Module.compile`) e devolve o próprio `model`.

    Usado pelo trainer e pelo `scripts/bench.py`. Ruling R25: compilar o modelo inteiro não adianta
    com grad checkpoint, porque o `torch.utils.checkpoint` desliga o Dynamo e o tronco todo rodava
    eager. Compilando cada Block do tronco e o do MTP, com o checkpoint por fora (`SoreLM.trunk`),
    só o kernel do FLA (`torch.compiler.disable`) fica eager. A `chunk_ce` (CE de cada chunk da
    loss, também sob checkpoint) é compilada junto: eager, ela materializa logits fp32 e log-probs
    `chunk x V` no backward (+1,2 GB no s150m a 2048). Os nomes dos parâmetros não mudam, então a
    divisão Muon/AdamW e o state_dict continuam os mesmos.

    `dynamic=False` mantém N (tokens do micro-batch) estático, porque o lowering do flex no Inductor
    falha com Q_LEN dinâmico. O nº de documentos (`len(cu_seqlens)`) é marcado dinâmico em
    `SoreLM.trunk`, para não recompilar a cada micro-batch.

    torch 2.14.1: a fusão "mix order reduction" do Inductor quebra no 1º backward do SoreLM
    (InductorError: AssertionError "509 v.s. 510" em MixOrderReduction.get_numel_rnumel).
    Desligada até o fix upstream; equivale a TORCHINDUCTOR_MIX_ORDER_REDUCTION=0.
    """
    import torch._inductor.config as inductor_config

    inductor_config.triton.mix_order_reduction = False
    modules = [*model.blocks, *([model.mtp.block] if model.mtp is not None else [])]
    if not model.cfg.fused_ce:
        modules.append(model.chunk_ce)
    for module in modules:
        module.compile(mode=mode, dynamic=False)
    return model


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()  # CPU: fp32


def _to_device(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(input_ids, cu_seqlens, labels)` na GPU. Em CUDA a origem é pinned e a cópia `non_blocking`.

    Cópia `non_blocking` de memória pageable é síncrona (o driver a copia para um buffer pinned
    antes); da pinned, a cópia só entra na fila da GPU e o host segue. Em CPU não há o que copiar.
    """
    tensors = (batch.input_ids, batch.cu_seqlens, batch.labels)
    if device.type == "cuda":
        tensors = tuple(t.pin_memory() for t in tensors)
    return tuple(t.to(device, non_blocking=True) for t in tensors)


def _train_step(
    model,
    optimizers,
    stream,
    cfg: TrainConfig,
    device,
    lr_mult: float,
    mtp_weight: float,
    grad_accum: int | None = None,
) -> dict:
    """Um passo de otimização; devolve as métricas do passo (médias ponderadas por token).

    O loss de cada micro-batch é `lm_loss + mtp_weight * mtp_loss`. O `out.loss` do modelo (que usa
    o `mtp_weight` fixo do ModelConfig) é ignorado.

    Entrada (D3): os `grad_accum` batches são copiados para a GPU de uma vez (pinned, `non_blocking`)
    antes do 1º forward, e o `cu_seqlens` da CPU vai junto ao modelo (`cu_seqlens_cpu`). Assim nada
    dentro do loop de micro-batches espera a CPU: a contagem de tokens válidos de cada um já está na
    CPU (`n_valid`) e o `out.n_tokens` do modelo, um tensor, não é lido aqui.
    """
    accum = grad_accum if grad_accum is not None else cfg.grad_accum
    batches = [next(stream) for _ in range(accum)]
    n_valid = [int((b.labels != IGNORE_INDEX).sum()) for b in batches]
    n_step = sum(n_valid)
    inputs = [_to_device(b, device) for b in batches]
    # loss, lm_loss, mtp_loss; passo sem nenhum token válido não tem loss (NaN -> null no log)
    sums = torch.zeros(3, device=device) if n_step else torch.full((3,), math.nan, device=device)
    for batch, (input_ids, cu_seqlens, labels), n in zip(batches, inputs, n_valid):
        if n == 0:
            continue
        weight = n / n_step
        with _autocast(device):
            out = model(input_ids, cu_seqlens=cu_seqlens, labels=labels, cu_seqlens_cpu=batch.cu_seqlens)
        if out.mtp_loss is None:
            loss, mtp = out.lm_loss, torch.full_like(out.lm_loss, math.nan)  # sem MTP -> null
        else:
            loss, mtp = out.lm_loss + mtp_weight * out.mtp_loss, out.mtp_loss
        (loss * weight).backward()
        sums += weight * torch.stack([t.detach() for t in (loss, out.lm_loss, mtp)])

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


# Sinais repetidos dentro desse intervalo contam como um só pedido de parada: com `uv run`, um
# Ctrl-C chega duas vezes ao Python (o SIGINT do terminal e o repassado pelo uv).
_SIGNAL_REPEAT_S = 1.0


@contextlib.contextmanager
def _stop_on_signals():
    """Durante o bloco, SIGINT/SIGTERM só pedem parada: devolve a lista que recebe o nº do sinal.

    Um 2º sinal (depois de `_SIGNAL_REPEAT_S`) reinstala os handlers anteriores e tem o efeito
    padrão: Ctrl-C de novo interrompe na hora, SIGTERM de novo encerra. Ao sair do bloco os
    handlers anteriores voltam. Fora da thread principal (onde o Python não aceita handlers) nada
    é instalado.
    """
    stop: list[int] = []
    if threading.current_thread() is not threading.main_thread():
        yield stop
        return
    signums = (signal.SIGINT, signal.SIGTERM)
    previous = {s: signal.getsignal(s) for s in signums}
    first_at = 0.0

    def restore() -> None:
        for s, handler in previous.items():
            signal.signal(s, signal.SIG_DFL if handler is None else handler)

    def request_stop(signum, frame) -> None:
        nonlocal first_at
        now = time.monotonic()
        if not stop:
            stop.append(signum)
            first_at = now
        elif now - first_at > _SIGNAL_REPEAT_S:
            restore()
            signal.raise_signal(signum)  # agora com o handler anterior

    for s in signums:
        signal.signal(s, request_stop)
    try:
        yield stop
    finally:
        restore()


def _crossed(before: int, after: int, every: int) -> bool:
    """True se `[before, after)` cruzou um múltiplo de `every` (`every = 0` desliga)."""
    return every > 0 and after // every > before // every


def train(cfg: TrainConfig, resume: Path | None = None, max_steps: int | None = None) -> TrainSummary:
    """Treina até `total_steps` ou até `max_steps` passos nesta chamada (o que vier antes).

    `resume` restaura modelo, otimizadores, RNG, cursor de dados e contadores de um `last.pt`. Sem
    `resume`, recusa um `out_dir` que já tem `last.pt` (FileExistsError). SIGINT/SIGTERM param o
    treino na fronteira do passo, com `last.pt` salvo, e `train` retorna normalmente.
    """
    device = resolve_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    if resume is None and (out_dir / "last.pt").exists():
        raise FileExistsError(
            f"{out_dir / 'last.pt'} já existe: para continuar esse treino use --resume {out_dir / 'last.pt'}; "
            "para um treino novo, use outro out_dir"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = SoreLM(get_preset(cfg.model_preset, **cfg.model_overrides)).to(device)
    optimizers = build_optimizers(model, cfg.lr_muon, cfg.lr_adamw, cfg.weight_decay)
    if cfg.compile:
        compile_model(model, mode=cfg.compile_mode)  # in-place, por Block: os nomes dos parâmetros não mudam
    first = cfg.stages[0]
    stream = PackedTokenStream(
        resolve_data_paths(cfg.data_paths), first.seq_len, first.micro_batch, cfg.eos_id, cfg.seed,
        unit_len=max(s.seq_len for s in cfg.stages),  # R22: a troca de estágio não reamostra o corpus
    )

    step, skipped, last_loss = 0, 0, math.nan
    spikes = LossSpikeDetector(cfg.spike_threshold)
    if resume is not None:
        data_state, train_state = load_checkpoint(resume, model, optimizers)
        for change in apply_config_hparams(optimizers, cfg.lr_muon, cfg.lr_adamw, cfg.weight_decay):
            print(f"[resume] aviso: {change} (vale o config)", flush=True)
        stream.load_state_dict(data_state)
        step, skipped, last_loss = train_state["step"], train_state["skipped_steps"], train_state["last_loss"]
        spikes.ema = train_state["loss_ema"]
    mtp_start = model.cfg.mtp_weight if cfg.mtp_weight_start is None else cfg.mtp_weight_start

    def checkpoint(snapshot: bool = False) -> None:
        """`last.pt` e, com `snapshot`, também `snapshot_step{step}.pt` (o mesmo conteúdo).

        Ficam só os `keep_snapshots` snapshots mais novos (o nome com 8 dígitos ordena por passo).
        """
        train_state = {
            "step": step,
            "skipped_steps": skipped,
            "tokens": stream.tokens_consumed,
            "last_loss": last_loss,
            "loss_ema": spikes.ema,
            "config": cfg.to_dict(),
        }
        paths = [out_dir / "last.pt"] + ([out_dir / f"snapshot_step{step:08d}.pt"] if snapshot else [])
        for path in paths:
            save_checkpoint(path, model, optimizers, stream.state_dict(), train_state)
        if snapshot:
            print(f"[ckpt] snapshot {paths[1].name}", flush=True)
            for old in sorted(out_dir.glob("snapshot_step*.pt"))[: -cfg.keep_snapshots]:
                old.unlink()

    total_steps = cfg.total_steps
    print(
        f"[train] {cfg.model_preset} {model.num_params() / 1e6:.1f}M params | {device} | "
        f"{total_steps} passos x {cfg.tokens_per_step} tokens | início no passo {step}",
        flush=True,
    )
    attempts, saved_at = 0, None
    with _stop_on_signals() as stop, open(out_dir / "metrics.jsonl", "a") as log:
        while step < total_steps and (max_steps is None or attempts < max_steps) and not stop:
            if _sync_stage(stream, cfg.stages):
                print(f"[stage] tokens={stream.tokens_consumed}: seq_len={stream.seq_len} "
                      f"micro_batch={stream.micro_batch}", flush=True)
            t0, tokens_before = time.perf_counter(), stream.tokens_consumed
            lr_mult = wsd_multiplier(step, cfg.warmup_steps, total_steps, cfg.decay_frac)
            mtp_weight = linear_schedule(step, total_steps, mtp_start, cfg.mtp_weight_end)
            stage = active_stage(cfg.stages, stream.tokens_consumed)
            stats = _train_step(
                model, optimizers, stream, cfg, device, lr_mult, mtp_weight, grad_accum=cfg.stage_grad_accum(stage)
            )
            ema = spikes.ema  # antes deste passo, para o aviso
            spike = step >= cfg.warmup_steps and spikes(stats["loss"])  # `step` ainda é o índice 0-based
            step, attempts = step + 1, attempts + 1
            skipped += stats["skipped"]
            last_loss = stats["loss"]
            tok_s = (stream.tokens_consumed - tokens_before) / (time.perf_counter() - t0)
            lr = optimizers[0].param_groups[0]["lr"]
            mem_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
            mfu = flops_per_token(model.cfg, stream.seq_len) * tok_s / (cfg.peak_tflops * 1e12)

            record = {
                "step": step,
                "tokens": stream.tokens_consumed,
                **{k: _json_float(stats[k]) for k in ("loss", "lm_loss", "mtp_loss", "grad_norm")},
                "lr_muon": lr,
                "mtp_weight": mtp_weight,
                "tok_s": tok_s,
                "mfu": mfu,
                "mem_gb": mem_gb,
                "seq_len": stream.seq_len,
                "skipped": stats["skipped"],
                "spike": spike,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if spike:
                print(f"[spike] passo {step}: loss {stats['loss']:.4f} > {1 + cfg.spike_threshold:g} × EMA "
                      f"{ema:.4f}", flush=True)
            if step % cfg.log_every == 0 or stats["skipped"]:
                print(
                    f"step {step}/{total_steps} | loss {stats['loss']:.4f} (lm {stats['lm_loss']:.4f} "
                    f"mtp {stats['mtp_loss']:.4f}) | gnorm {stats['grad_norm']:.3f} | lr {lr:.2e} | "
                    f"{tok_s:,.0f} tok/s | MFU {mfu:.1%} | {mem_gb:.2f} GB | seq {stream.seq_len}"
                    + (" | SKIPPED (grad norm não finito)" if stats["skipped"] else ""),
                    flush=True,
                )
            snapshot = _crossed(tokens_before, stream.tokens_consumed, cfg.snapshot_every_tokens)
            if snapshot or _crossed(tokens_before, stream.tokens_consumed, cfg.ckpt_every_tokens):
                checkpoint(snapshot)
                saved_at = step
        if saved_at != step:
            checkpoint()
        if stop:
            print(f"[train] {signal.Signals(stop[0]).name}: parada no passo {step} com last.pt salvo; "
                  f"retome com --resume {out_dir / 'last.pt'}", flush=True)

    summary = TrainSummary(steps=step, tokens=stream.tokens_consumed, skipped_steps=skipped, last_loss=last_loss)
    print(f"[train] fim: {summary}", flush=True)
    return summary
