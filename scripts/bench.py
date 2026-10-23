"""Benchmark de throughput e memória de um passo de treino completo (fwd + bwd + otimizadores).

Monta o preset na GPU (backends de GDN e atenção do próprio preset) e roda `--steps` passos sobre
micro-batches empacotados sintéticos: `seq_len * micro_batch` tokens em `[1, N]`, com 4 documentos
por janela (fronteiras sorteadas a cada passo, mesmo número de documentos para não mudar shapes).
Cada passo é o do trainer: `--grad-accum` micro-batches (autocast bf16, forward com labels e
backward), clip do grad norm e um passo do Muon + AdamW. O passo do otimizador (Newton-Schulz do
Muon) pesa muito com `--grad-accum 1`; no treino real ele se dilui em 16-64 micro-batches. Mede a
mediana de tok/s dos passos após o `--warmup`, o MFU via `flops_per_token` e o pico de VRAM
(`torch.cuda.max_memory_allocated`).

Uso: uv run python scripts/bench.py --preset s150m --seq-len 2048 --micro-batch 1 --steps 10 \
         --grad-checkpoint --peak-tflops 35 [--compile] [--grad-accum 1] [--warmup 2] [--json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch
from torch import Tensor

from sore.model import SoreLM, get_preset
from sore.model.flops import flops_per_token
from sore.model.loss import IGNORE_INDEX
from sore.train.optim import build_optimizers
from sore.train.trainer import compile_model

DOCS_PER_WINDOW = 4


def synthetic_batch(
    seq_len: int, micro_batch: int, vocab: int, gen: torch.Generator
) -> tuple[Tensor, Tensor, Tensor]:
    """`input_ids` e `labels` `[1, N]` aleatórios e `cu_seqlens` com `DOCS_PER_WINDOW` documentos por janela."""
    n = seq_len * micro_batch
    ids = torch.randint(0, vocab, (1, n), generator=gen)
    labels = torch.randint(0, vocab, (1, n), generator=gen)
    n_cuts = min(DOCS_PER_WINDOW, seq_len) - 1
    cuts = [0]
    for w in range(micro_batch):
        inner = (torch.randperm(seq_len - 1, generator=gen)[:n_cuts] + 1).sort().values  # em [1, seq_len)
        cuts += (w * seq_len + inner).tolist() + [(w + 1) * seq_len]
    cu_seqlens = torch.tensor(cuts, dtype=torch.long)
    labels[0, cu_seqlens[1:] - 1] = IGNORE_INDEX  # como no stream: a posição do EOS não tem alvo
    return ids, labels, cu_seqlens


def train_step(
    fwd, model: SoreLM, optimizers: list[torch.optim.Optimizer], batches: list[tuple[Tensor, ...]]
) -> None:
    """Um passo do trainer: backward de cada micro-batch (loss / n), clip em 1.0 e passo dos otimizadores."""
    for ids, labels, cu_seqlens in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fwd(ids, cu_seqlens=cu_seqlens, labels=labels)
        (out.loss / len(batches)).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for opt in optimizers:
        opt.step()
        opt.zero_grad(set_to_none=True)


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
    parser.add_argument("--compile", action="store_true", help="torch.compile no modelo (como o trainer)")
    parser.add_argument("--grad-checkpoint", action="store_true", help="activation checkpointing por bloco")
    parser.add_argument("--peak-tflops", type=float, default=165.0, help="pico BF16 denso da GPU (4090: 165)")
    parser.add_argument("--json", action="store_true", help="imprime também o resultado em JSON")
    args = parser.parse_args()
    if min(args.seq_len, args.micro_batch, args.grad_accum) < 1 or args.peak_tflops <= 0:
        parser.error("--seq-len, --micro-batch, --grad-accum e --peak-tflops devem ser positivos")
    if not 0 <= args.warmup < args.steps:
        parser.error("exige 0 <= --warmup < --steps")
    if not torch.cuda.is_available():
        parser.error("o benchmark exige GPU CUDA")

    device = torch.device("cuda")
    torch.manual_seed(0)
    cfg = get_preset(args.preset, grad_checkpoint=args.grad_checkpoint)
    model = SoreLM(cfg).to(device)
    optimizers = build_optimizers(model, lr_muon=0.02, lr_adamw=3e-3, weight_decay=0.1)
    fwd = compile_model(model) if args.compile else model
    flops = flops_per_token(cfg, args.seq_len)
    tokens = args.seq_len * args.micro_batch * args.grad_accum  # por passo
    print(
        f"[bench] {args.preset}: {model.num_params() / 1e6:.1f}M params | {flops / 1e9:.2f} GFLOP/token | "
        f"gdn={cfg.gdn_backend} attn={cfg.attn_backend} | {torch.cuda.get_device_name(device)}",
        flush=True,
    )

    gen = torch.Generator().manual_seed(0)
    times: list[float] = []
    for step in range(args.steps):
        batches = [
            tuple(t.to(device) for t in synthetic_batch(args.seq_len, args.micro_batch, cfg.vocab_size, gen))
            for _ in range(args.grad_accum)
        ]
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        try:
            train_step(fwd, model, optimizers, batches)
            torch.cuda.synchronize(device)
        except torch.OutOfMemoryError:
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"[bench] OOM no passo {step + 1}: pico alocado {peak:.2f} GB", flush=True)
            sys.exit(1)
        times.append(time.perf_counter() - t0)
        print(f"  passo {step + 1}/{args.steps}: {times[-1]:.3f} s ({tokens / times[-1]:,.0f} tok/s)", flush=True)

    step_s = statistics.median(times[args.warmup :])
    tok_s = tokens / step_s
    result = {
        "preset": args.preset,
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
    }
    print(
        f"[bench] {args.preset} seq {args.seq_len} x mb {args.micro_batch} x accum {args.grad_accum} | "
        f"compile={'on' if args.compile else 'off'} ckpt={'on' if args.grad_checkpoint else 'off'} | "
        f"{tok_s:,.0f} tok/s (mediana de {args.steps - args.warmup}) | "
        f"MFU {result['mfu']:.1%} (pico {args.peak_tflops:g} TFLOPs) | "
        f"VRAM pico {result['mem_gb']:.2f} GB (reservado {result['mem_reserved_gb']:.2f} GB)",
        flush=True,
    )
    if args.json:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
