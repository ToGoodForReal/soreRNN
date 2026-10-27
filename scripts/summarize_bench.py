"""Sumariza os resultados do benchmark gerados por scripts/bench.py (--out FILE.jsonl).

Lê cada linha JSON, lida com entradas normais e entradas de OOM, e gera uma tabela em
Markdown ordenada por tok/s (decrescente, OOM no fim).
Destaca configurações que atingem a meta de 15.000 tok/s (MFU >= 50%) ou o piso de 11.800 tok/s.

Uso:
    python scripts/summarize_bench.py runs/day1/bench.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def format_config_desc(entry: dict) -> str:
    """Gera uma descrição legível da configuração e overrides."""
    parts = []
    preset = entry.get("preset", "")
    seq_len = entry.get("seq_len")
    mb = entry.get("micro_batch")
    shape = f"{seq_len}x{mb}" if seq_len and mb else ""
    if preset and shape:
        parts.append(f"{preset} {shape}")
    elif preset:
        parts.append(preset)
    elif shape:
        parts.append(shape)

    overrides = entry.get("overrides") or {}
    if overrides:
        ov_strs = [f"{k}={v}" for k, v in sorted(overrides.items())]
        parts.append(", ".join(ov_strs))

    extras = []
    if entry.get("compile"):
        extras.append("compile")
    if entry.get("fp8") and entry.get("fp8") != "off":
        extras.append(f"fp8={entry['fp8']}")
    if entry.get("grad_checkpoint"):
        extras.append("ckpt")
    if extras:
        parts.append(f"[{' '.join(extras)}]")

    return " | ".join(parts) if parts else (preset or "base")


def format_table(entries: list[dict]) -> str:
    """Formata a lista de resultados em uma tabela Markdown ordenada."""
    normal_entries = [e for e in entries if not e.get("oom", False)]
    oom_entries = [e for e in entries if e.get("oom", False)]

    # Ordena decrescente por throughput (tok/s); OOM fica no final
    normal_entries.sort(key=lambda e: float(e.get("tok_s", 0.0)), reverse=True)
    sorted_entries = normal_entries + oom_entries

    lines = [
        "| Config / Overrides | tok/s | MFU | VRAM Peak/Reserved (GB) | Flags |",
        "|---|---|---|---|---|",
    ]

    for entry in sorted_entries:
        desc = format_config_desc(entry)
        if entry.get("oom", False):
            tok_str = "OOM"
            mfu_str = "-"
            mem_peak = entry.get("mem_gb")
            vram_str = f"{mem_peak:.2f} GB" if mem_peak is not None else "-"
            flag_str = "**OOM**"
        else:
            tok_s = float(entry.get("tok_s", 0.0))
            mfu = float(entry.get("mfu", 0.0))
            tok_str = f"{tok_s:,.0f}"
            mfu_str = f"{mfu:.1%}"

            mem_peak = entry.get("mem_gb")
            mem_res = entry.get("mem_reserved_gb")
            if mem_peak is not None and mem_res is not None:
                vram_str = f"{mem_peak:.2f} / {mem_res:.2f}"
            elif mem_res is not None:
                vram_str = f"{mem_res:.2f}"
            elif mem_peak is not None:
                vram_str = f"{mem_peak:.2f}"
            else:
                vram_str = "-"

            if entry.get("finite_grads") is False:
                flag_str = "**INVÁLIDO (grad norm NaN)**"
            elif tok_s >= 15000.0:
                flag_str = ">= 15.000 tok/s (meta 50%)"
            elif tok_s >= 11800.0:
                flag_str = ">= 11.800 tok/s (piso)"
            else:
                flag_str = "< 11.800 tok/s"

        lines.append(f"| {desc} | {tok_str} | {mfu_str} | {vram_str} | {flag_str} |")

    return "\n".join(lines)


def summarize_bench(bench_path: Path | str) -> str:
    """Lê o arquivo jsonl do benchmark e retorna a tabela markdown formatada."""
    path = Path(bench_path)
    entries: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return format_table(entries)


def find_best_exact(bench_path: Path | str, seq_len: int, micro_batch: int) -> dict | None:
    """Encontra a melhor configuração exata (maior tok/s, sem OOM, sem FP8) para uma sequência."""
    path = Path(bench_path)
    if not path.exists():
        return None
    candidates: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("oom", False) or not entry.get("finite_grads", True):
                continue
            if entry.get("fp8", "off") != "off":
                continue
            if entry.get("seq_len") == seq_len and entry.get("micro_batch") == micro_batch:
                candidates.append(entry)
    if not candidates:
        return None
    candidates.sort(key=lambda e: float(e.get("tok_s", 0.0)), reverse=True)
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bench_file", type=Path, help="caminho para o bench.jsonl")
    parser.add_argument(
        "--best-exact",
        nargs=2,
        type=int,
        metavar=("SEQ_LEN", "MICRO_BATCH"),
        help="imprime os argumentos --override da melhor configuração exata para seq_len e micro_batch",
    )
    args = parser.parse_args(argv)

    if not args.bench_file.exists():
        print(f"Erro: arquivo não encontrado: {args.bench_file}", file=sys.stderr)
        return 1

    if args.best_exact:
        seq_len, mb = args.best_exact
        best = find_best_exact(args.bench_file, seq_len, mb)
        if best:
            ovs = best.get("overrides", {})
            args_list = [f"--override {k}={v}" for k, v in sorted(ovs.items())]
            print(" ".join(args_list))
        return 0

    table = summarize_bench(args.bench_file)
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
