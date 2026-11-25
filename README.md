# soreRNN

soreRNN v2: modelo de linguagem recorrente moderno (arquitetura "soreQ4", estilo Qwen4, foco em
PT-BR). Camadas Gated DeltaNet intercaladas com atenção global com gate, FFN SwiGLU densa e MTP.
Implementado em PyTorch (BF16) com kernels Triton do
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention).

O alvo atual é o preset `b0_4` (~0,41B parâmetros), pré-treinado em ~10B tokens numa RTX 3050 8GB, com
o tokenizer do Qwen3.5 podado para 98.304 ids. A operação completa (tokenizer, dados, portões,
lançamento e monitoramento) está em [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

| Preset | Parâmetros | Uso |
|---|---:|---|
| `tiny` | ~1M | testes e smoke local |
| `s150m` | ~0,16B (vocab 65.536) | portões de qualidade e ablações |
| `b0_4` | ~0,41B | pré-treino na RTX 3050 8GB |
| `b0_8` | ~0,82B | plano original na RTX 4090, suspenso ([`docs/history/`](docs/history/)) |

Design: `docs/superpowers/specs/2026-10-23-sorernn-v2-design.md`. Planos de implementação em
`docs/superpowers/plans/`.

## Setup (Linux nativo ou WSL2 + GPU NVIDIA)

Requer Python 3.12 e [uv](https://docs.astral.sh/uv/). As versões ficam travadas em `uv.lock` (torch
2.14.1 cu130, triton 3.8.0, flash-linear-attention 0.5.2).

```bash
uv sync
bash scripts/setup_worker.sh --check   # driver, uv, ambiente, torch com CUDA e fla
```

No WSL2, com o checkout em `/mnt/c/...`, mantenha o venv no filesystem do WSL:
`export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/sorernn` antes do `uv sync`.

## Testes

```bash
uv run pytest -q
```

A suíte passa em CPU, sem Triton e sem FLA. Testes marcados com `@pytest.mark.cuda`, `@pytest.mark.fla`
ou `@pytest.mark.fp8` são pulados automaticamente quando a GPU, o FLA ou o FP8 (sm89+) não estão
disponíveis. Na RTX 3050 (sm86) os testes `@fp8` são sempre pulados.

## Estrutura

- `sore/model/`: o modelo (`SoreLM`, presets em `config.py`, FLOPs em `flops.py`).
- `sore/train/`: trainer, otimizadores (Muon + AdamW), stream de dados, checkpoints, FP8 e o pipeline de
  dados do pré-treino (`sources.py`, `pretrain_data.py`, `tokenizer_prune.py`).
- `scripts/`: poda do tokenizer, geração de dados, benchmark, lançador do pré-treino e campanha de qualidade.
- `configs/`: configs de treino (YAML), carregados por `python -m sore.train --config ...`.

## Motor C++/CUDA (legado)

O motor C++20/CUDA original (tensor core, autograd, LinearRNN, cuBLAS, scan paralelo, AdamW fundido)
está congelado em `legacy/cpp/` e não recebe mais features. Para compilar:
`cmake -S legacy/cpp -B build && cmake --build build -j`. O estado exato da base está marcado pela tag
`v0.1-cpp-engine`; o README original (build, testes e exemplos) está em
`git show v0.1-cpp-engine:README.md`.

## License

MIT
