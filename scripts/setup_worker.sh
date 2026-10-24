#!/usr/bin/env bash
# Prepara uma máquina "worker" (Linux nativo ou WSL) para rodar a campanha de qualidade.
#
# Uso: bash scripts/setup_worker.sh [--check]
#   (sem flag)  confere a GPU, instala o uv se faltar, `uv sync` e os testes @cuda/@fla
#   --check     só valida (GPU, uv, ambiente já sincronizado); não instala nem baixa nada
#
# Pressupõe o repositório já presente (git bundle ou push da branch; ver docs/RUNBOOK_4090.md,
# "Segunda máquina") e a amostra PT em data/sample_pt/. Termina com 0 se tudo está pronto.
set -euo pipefail
cd "$(dirname "$0")/.."

CHECK=0
case "${1:-}" in
  "") ;;
  --check) CHECK=1 ;;
  *) echo "uso: $0 [--check]" >&2; exit 2 ;;
esac
export PATH="$HOME/.local/bin:$PATH"  # onde o instalador do uv coloca o binário

fail() { echo "[setup] ERRO: $*" >&2; exit 1; }

# 1. GPU: o driver precisa expor CUDA >= 13.0 (os wheels torch 2.14.1 do uv.lock são cu130)
command -v nvidia-smi >/dev/null || fail "nvidia-smi não encontrado (driver NVIDIA ausente; no WSL, driver do Windows)"
smi=$(nvidia-smi)  # em variável: com pipefail, um `head` fechando o pipe daria SIGPIPE
# drivers novos escrevem "CUDA UMD Version: 13.4" no lugar de "CUDA Version: 13.4"
cuda=$(sed -n 's/.*CUDA \(UMD \)\{0,1\}Version: *\([0-9]*\.[0-9]*\).*/\2/p' <<< "$smi" | head -n 1)
[[ -n "$cuda" ]] || fail "não consegui ler 'CUDA Version' da saída do nvidia-smi"
major=${cuda%%.*}
[[ "$major" -ge 13 ]] || fail "driver com CUDA $cuda; é preciso >= 13.0 (atualize o driver NVIDIA)"
echo "[setup] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1) | CUDA $cuda"

# 2. uv
if ! command -v uv >/dev/null; then
  [[ $CHECK -eq 0 ]] || fail "uv não encontrado (rode sem --check para instalar)"
  echo "[setup] instalando o uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "[setup] $(uv --version)"

# 3. dados e dependências
[[ -f data/sample_pt/sample_000.bin ]] || fail "data/sample_pt/sample_000.bin ausente (copie ou gere a amostra, RUNBOOK §8)"
if [[ $CHECK -eq 1 ]]; then
  uv sync --frozen --check >/dev/null 2>&1 || fail "ambiente desatualizado em relação ao uv.lock (rode sem --check)"
  echo "[setup] --check: GPU, uv, amostra e ambiente OK"
  exit 0
fi
uv sync --frozen

# 4. testes que exigem GPU/FLA têm de EXECUTAR (não ser pulados)
uv run pytest -q -m "cuda or fla"
echo "[setup] pronto"
