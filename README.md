# soreRNN

soreRNN v2: modelo de linguagem recorrente moderno (arquitetura "soreQ4", estilo
Qwen4, ~0.8B parâmetros, foco em PT-BR). Camadas Gated DeltaNet intercaladas com
atenção global com gate, FFN SwiGLU densa e MTP. Implementado em PyTorch (BF16)
com kernels Triton do [flash-linear-attention](https://github.com/fla-org/flash-linear-attention).

Design completo: `docs/superpowers/specs/2026-10-23-sorernn-v2-design.md`.
Plano de implementação: `docs/superpowers/plans/2026-10-23-sorernn-v2-core.md`.

## Setup (WSL2 Ubuntu 24.04 + GPU NVIDIA)

Requer Python 3.12 e [uv](https://docs.astral.sh/uv/). As versões ficam travadas
em `uv.lock` (torch 2.14.1 cu130, triton 3.8.0, flash-linear-attention 0.5.2).

```bash
# o checkout fica em /mnt/c/...; mantenha o venv no filesystem do WSL
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/sorernn
uv sync
```

## Testes

```bash
uv run pytest -q
```

A suíte passa em CPU, sem Triton e sem FLA. Testes marcados com `@pytest.mark.cuda`
ou `@pytest.mark.fla` são pulados automaticamente quando a GPU ou o FLA não estão
disponíveis.

## Motor C++/CUDA (legado)

O motor C++20/CUDA original (tensor core, autograd, LinearRNN, cuBLAS, scan
paralelo, AdamW fundido) está congelado em `legacy/cpp/` e não recebe mais
features. Para compilar: `cmake -S legacy/cpp -B build && cmake --build build -j`.
O estado exato da base está marcado pela tag `v0.1-cpp-engine`; o README original
(build, testes e exemplos) está em `git show v0.1-cpp-engine:README.md`.

## License

MIT
