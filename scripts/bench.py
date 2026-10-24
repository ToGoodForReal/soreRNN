"""Benchmark de throughput e memória de um passo de treino completo (fwd + bwd + otimizadores).

Monta o preset na GPU (backends de GDN e atenção do próprio preset, mais os `--override`) e roda
`--steps` passos sobre micro-batches empacotados sintéticos: `seq_len * micro_batch` tokens em
`[1, N]`. Sem `--doc-lengths`, cada janela tem 4 documentos (fronteiras sorteadas a cada passo, mesmo
número de documentos para não mudar shapes). Com `--doc-lengths` (o `doc_lengths.json` da amostra PT),
os comprimentos são sorteados dessa distribuição real, janela a janela (um documento nunca atravessa
a janela), então o número de documentos varia entre micro-batches, como no treino.
Cada passo é o do trainer: monta os `--grad-accum` micro-batches e copia todos para a GPU de uma
vez, pinned e `non_blocking`, antes do 1º forward (fase `data`), faz o forward com labels e o
backward de cada um (autocast bf16, `cu_seqlens_cpu` junto, fase `fwd_bwd`), o clip do grad norm
(`clip`) e um passo do Muon + AdamW (`optimizer`). O passo do otimizador (Newton-Schulz do Muon)
pesa muito com `--grad-accum 1`; no treino real ele se dilui em 16-64 micro-batches. Mede a mediana
de tok/s dos passos após o `--warmup` (o tempo do passo inclui a fase `data`), o MFU via
`flops_per_token` e o pico de VRAM (`torch.cuda.max_memory_allocated`). As fases são medidas com
eventos CUDA (mediana por passo, em segundos).

`--profile` roda ainda dois passos extras, depois dos medidos e com os mesmos batches (o 1º aquece,
o 2º fica sob o `torch.profiler`), e soma o tempo de GPU do 2º por categoria de kernel
(`sore.train.profiling`), mais `idle`: a parede desse passo menos a união dos intervalos de kernels
(a GPU esperando a CPU). O passo perfilado tem o overhead do profiler e não inclui a montagem do batch.

`--no-cpu-cu-seqlens` não passa o `cu_seqlens_cpu` ao modelo: o FLA volta a ler o `cu_seqlens` da GPU
no host (um sync por micro-batch no `prepare_chunk_indices`). Serve para medir o ganho de não sincronizar.

`--out FILE.jsonl` acrescenta uma linha JSON por execução (a mesma do `--json`).

Uso: uv run python scripts/bench.py --preset s150m --seq-len 2048 --micro-batch 1 --steps 10 \
         --grad-checkpoint --peak-tflops 35 [--compile] [--grad-accum 1] [--warmup 2] [--json] \
         [--override n_layers=4] [--doc-lengths data/sample_pt/doc_lengths.json] [--profile] [--out F.jsonl]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from sore.model import SoreLM, get_preset
from sore.model.flops import flops_per_token
from sore.model.loss import IGNORE_INDEX
from sore.train.config import parse_overrides
from sore.train.optim import build_optimizers
from sore.train.profiling import idle_seconds, kernel_breakdown
from sore.train.text_sample import sample_doc_lengths
from sore.train.trainer import compile_model

DOCS_PER_WINDOW = 4
PHASES = ("data", "fwd_bwd", "clip", "optimizer")


def synthetic_batch(
    seq_len: int,
    micro_batch: int,
    vocab: int,
    gen: torch.Generator,
    doc_lengths: Sequence[int] | np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """`input_ids` e `labels` `[1, N]` aleatórios e o `cu_seqlens` dos documentos.

    Sem `doc_lengths`, cada janela tem `DOCS_PER_WINDOW` documentos; com ele, os comprimentos de cada
    janela vêm de `sample_doc_lengths(doc_lengths, seq_len, rng)` e o número de documentos varia.
    """
    n = seq_len * micro_batch
    ids = torch.randint(0, vocab, (1, n), generator=gen)
    labels = torch.randint(0, vocab, (1, n), generator=gen)
    cuts = [0]
    for w in range(micro_batch):
        if doc_lengths is None:
            n_cuts = min(DOCS_PER_WINDOW, seq_len) - 1
            inner = (torch.randperm(seq_len - 1, generator=gen)[:n_cuts] + 1).sort().values  # em [1, seq_len)
            cuts += (w * seq_len + inner).tolist() + [(w + 1) * seq_len]
        else:
            assert rng is not None, "doc_lengths exige o gerador rng"
            cuts += (w * seq_len + np.cumsum(sample_doc_lengths(doc_lengths, seq_len, rng))).tolist()
    cu_seqlens = torch.tensor(cuts, dtype=torch.long)
    labels[0, cu_seqlens[1:] - 1] = IGNORE_INDEX  # como no stream: a posição do EOS não tem alvo
    return ids, labels, cu_seqlens


def train_step(
    model: SoreLM,
    optimizers: list[torch.optim.Optimizer],
    make_batches: Callable[[], list[tuple[Tensor, Tensor, Tensor, Tensor | None]]],
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Um passo do trainer; devolve `(parede_s, fases_s)`.

    Monta os micro-batches (`make_batches`: `(ids, labels, cu_seqlens, cu_seqlens_cpu)`, já na GPU e
    com o `cu_seqlens_cpu` na CPU, ou `None`), faz o backward de cada um (loss / n), o clip em 1.0 e o
    passo dos otimizadores. As fases `PHASES` saem de eventos CUDA gravados entre elas: com a GPU
    esperando a CPU, o tempo ocioso entra na fase em que ocorreu (as fases somam a parede).
    """
    marks = [torch.cuda.Event(enable_timing=True) for _ in range(len(PHASES) + 1)]
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    marks[0].record()
    batches = make_batches()
    marks[1].record()
    for ids, labels, cu_seqlens, cu_seqlens_cpu in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(ids, cu_seqlens=cu_seqlens, labels=labels, cu_seqlens_cpu=cu_seqlens_cpu)
        (out.loss / len(batches)).backward()
    marks[2].record()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    marks[3].record()
    for opt in optimizers:
        opt.step()
        opt.zero_grad(set_to_none=True)
    marks[4].record()
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - t0
    return wall, {p: marks[i].elapsed_time(marks[i + 1]) / 1e3 for i, p in enumerate(PHASES)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", required=True, help="preset do modelo (tiny, s150m, b0_8)")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True, help="passos totais, incluindo o warmup")
    parser.add_argument(
        "--grad-accum", type=int, default=1, help="micro-batches por passo do otimizador (o trainer usa 16-64)"
    )
    parser.add_argument("--warmup", type=int, default=2, help="passos iniciais fora da mediana")
    parser.add_argument("--compile", action="store_true", help="torch.compile por Block (compile_model, como o trainer)")
    parser.add_argument("--grad-checkpoint", action="store_true", help="activation checkpointing por bloco")
    parser.add_argument("--peak-tflops", type=float, default=165.0, help="pico BF16 denso da GPU (4090: 165)")
    parser.add_argument("--json", action="store_true", help="imprime também o resultado em JSON")
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="campo do ModelConfig do preset, valor em YAML (repetível); `--override n_layers=4` é a fatia do b0_8",
    )
    parser.add_argument(
        "--doc-lengths", type=Path, help="doc_lengths.json da amostra PT: sorteia os comprimentos de documento dele"
    )
    parser.add_argument(
        "--no-cpu-cu-seqlens", action="store_true",
        help="não passa o cu_seqlens_cpu ao modelo (o FLA sincroniza com a GPU a cada micro-batch)",
    )
    parser.add_argument("--profile", action="store_true", help="um passo extra no torch.profiler: tempo por categoria de kernel")
    parser.add_argument("--out", type=Path, help="acrescenta o resultado, em uma linha JSON, a este arquivo .jsonl")
    args = parser.parse_args()
    if min(args.seq_len, args.micro_batch, args.grad_accum) < 1 or args.peak_tflops <= 0:
        parser.error("--seq-len, --micro-batch, --grad-accum e --peak-tflops devem ser positivos")
    if not 0 <= args.warmup < args.steps:
        parser.error("exige 0 <= --warmup < --steps")
    if not torch.cuda.is_available():
        parser.error("o benchmark exige GPU CUDA")
    try:
        overrides = parse_overrides(args.override)
        cfg = get_preset(args.preset, **{"grad_checkpoint": args.grad_checkpoint, **overrides})
    except (ValueError, KeyError, TypeError) as e:
        parser.error(f"--preset/--override inválido: {e}")
    # Um array numpy, uma vez: o `np.asarray` do `sample_doc_lengths` vira de graça a cada janela e a
    # fase `data` mede só a montagem do batch e a cópia para a GPU, não converter a lista toda vez.
    doc_lengths = (
        np.asarray(json.loads(args.doc_lengths.read_text())["lengths"], dtype=np.int64) if args.doc_lengths else None
    )

    device = torch.device("cuda")
    torch.manual_seed(0)
    model = SoreLM(cfg).to(device)
    optimizers = build_optimizers(model, lr_muon=3e-3, lr_adamw=3e-3, weight_decay=0.1)  # defaults do TrainConfig
    if args.compile:
        compile_model(model)  # in-place, por Block
    flops = flops_per_token(cfg, args.seq_len)
    tokens = args.seq_len * args.micro_batch * args.grad_accum  # por passo
    print(
        f"[bench] {args.preset}: {model.num_params() / 1e6:.1f}M params | {flops / 1e9:.2f} GFLOP/token | "
        f"gdn={cfg.gdn_backend} attn={cfg.attn_backend} | {torch.cuda.get_device_name(device)}"
        + (f" | overrides {overrides}" if overrides else "")
        + (f" | doc-lengths {args.doc_lengths}" if args.doc_lengths else ""),
        flush=True,
    )

    gen = torch.Generator().manual_seed(0)
    doc_rng = np.random.default_rng(0)

    def make_batches() -> list[tuple[Tensor, Tensor, Tensor, Tensor | None]]:
        # como o trainer (D3): monta os `grad_accum` batches e copia todos (pinned, non_blocking) antes do 1º forward
        host = [
            synthetic_batch(args.seq_len, args.micro_batch, cfg.vocab_size, gen, doc_lengths, doc_rng)
            for _ in range(args.grad_accum)
        ]
        cu_cpu = [None if args.no_cpu_cu_seqlens else cu for _, _, cu in host]
        return [
            (*(t.pin_memory().to(device, non_blocking=True) for t in batch), cpu)
            for batch, cpu in zip(host, cu_cpu)
        ]

    times: list[float] = []
    phases: list[dict[str, float]] = []
    for step in range(args.steps):
        try:
            wall, step_phases = train_step(model, optimizers, make_batches, device)
        except torch.OutOfMemoryError:
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"[bench] OOM no passo {step + 1}: pico alocado {peak:.2f} GB", flush=True)
            sys.exit(1)
        times.append(wall)
        phases.append(step_phases)
        print(f"  passo {step + 1}/{args.steps}: {wall:.3f} s ({tokens / wall:,.0f} tok/s)", flush=True)

    step_s = statistics.median(times[args.warmup :])
    tok_s = tokens / step_s
    result = {
        "preset": args.preset,
        "overrides": overrides,
        "doc_lengths": str(args.doc_lengths) if args.doc_lengths else None,
        "cpu_cu_seqlens": not args.no_cpu_cu_seqlens,
        "seq_len": args.seq_len,
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "compile": args.compile,
        "grad_checkpoint": args.grad_checkpoint,
        "steps": args.steps,
        "warmup": args.warmup,
        "step_s": step_s,
        "tok_s": tok_s,
        "flops_per_token": flops,
        "peak_tflops": args.peak_tflops,
        "mfu": flops * tok_s / (args.peak_tflops * 1e12),
        "mem_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "mem_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
        "phases": {p: statistics.median(s[p] for s in phases[args.warmup :]) for p in PHASES},
    }
    if args.profile:
        # Os mesmos batches duas vezes: um micro-batch com formato inédito (por exemplo, janela de 1
        # documento só) recompila os Blocks na hora, e esse passo viraria só "idle". O 1º passo
        # (sem profiler) aquece, o 2º é o perfilado; por isso a fase `data` dele é desprezível.
        fixed = make_batches()
        train_step(model, optimizers, lambda: fixed, device)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            prof_wall, _ = train_step(model, optimizers, lambda: fixed, device)
        result["kernels"] = {**kernel_breakdown(prof), "idle": idle_seconds(prof, prof_wall)}
    print(
        f"[bench] {args.preset} seq {args.seq_len} x mb {args.micro_batch} x accum {args.grad_accum} | "
        f"compile={'on' if args.compile else 'off'} ckpt={'on' if args.grad_checkpoint else 'off'} | "
        f"{tok_s:,.0f} tok/s (mediana de {args.steps - args.warmup}) | "
        f"MFU {result['mfu']:.1%} (pico {args.peak_tflops:g} TFLOPs) | "
        f"VRAM pico {result['mem_gb']:.2f} GB (reservado {result['mem_reserved_gb']:.2f} GB)",
        flush=True,
    )
    print("[bench] fases (s/passo): " + " | ".join(f"{p} {v:.3f}" for p, v in result["phases"].items()), flush=True)
    if args.profile:
        print(
            f"[bench] GPU por categoria no passo perfilado ({prof_wall:.3f} s): "
            + " | ".join(f"{c} {v:.3f} s ({v / prof_wall:.0%})" for c, v in result["kernels"].items()),
            flush=True,
        )
    if args.json:
        print(json.dumps(result))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
